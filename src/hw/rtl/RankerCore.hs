{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BinaryLiterals #-}

module RankerCore where

import Clash.Prelude
import qualified Prelude as P

type Fix16 = SFixed 16 16

type NumHashLanes = 128
type TopKDepth = 16
type BitmaskWords = 2
type TokenSetSize = 64
type CandTokensSize = 16

fixWeightBase :: Fix16
fixWeightBase = 0.25

fixWeightOverlap :: Fix16
fixWeightOverlap = 0.1875

fixWeightJaccard :: Fix16
fixWeightJaccard = 0.1875

fixWeightDiversity :: Fix16
fixWeightDiversity = 0.1875

fixWeightProximity :: Fix16
fixWeightProximity = 0.1875

fixMaxRawScoreInv :: Fix16
fixMaxRawScoreInv = 0.010009765625

hashMultA :: Unsigned 64
hashMultA = 0x9e3779b97f4a7c15

hashMultB :: Unsigned 64
hashMultB = 0x517cc1b727220a95

newtype ScoreFix = ScoreFix { unScoreFix :: Fix16 }
    deriving stock (Generic, Eq, Ord, Show)
    deriving newtype (NFDataX, ShowX, BitPack)

newtype SegmentID = SegmentID { unSegmentID :: Unsigned 64 }
    deriving stock (Generic, Eq, Ord, Show)
    deriving newtype (NFDataX, ShowX, BitPack)

newtype Position = Position { unPosition :: Unsigned 64 }
    deriving stock (Generic, Eq, Ord, Show)
    deriving newtype (NFDataX, ShowX, BitPack)

newtype Token = Token { unToken :: Unsigned 32 }
    deriving stock (Generic, Eq, Ord, Show)
    deriving newtype (NFDataX, ShowX, BitPack)

data TokenSet = TokenSet
    { tsEntries :: Vec TokenSetSize (Maybe Token)
    , tsCount   :: Index (TokenSetSize + 1)
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

initTokenSet :: TokenSet
initTokenSet = TokenSet (repeat Nothing) 0

containsToken :: TokenSet -> Token -> Bool
containsToken (TokenSet entries _) tok =
    foldl (\acc e -> acc || e == Just tok) False entries

insertToken :: TokenSet -> Token -> (TokenSet, Bool)
insertToken (TokenSet entries cnt) tok =
    if containsToken (TokenSet entries cnt) tok
    then (TokenSet entries cnt, False)
    else if cnt == maxBound
         then (TokenSet entries cnt, False)
         else
             let idx = resize cnt :: Index TokenSetSize
                 newEntries = replace idx (Just tok) entries
                 newCnt = cnt + 1
             in (TokenSet newEntries newCnt, True)

data TokenStreamIn = TokenStreamIn
    { tsToken     :: Token
    , tsValid     :: Bool
    , tsLast      :: Bool
    , tsAnchorPos :: Position
    , tsHasAnchor :: Bool
    , tsSegmentID :: SegmentID
    , tsPosition  :: Position
    , tsBaseScore :: ScoreFix
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

data CandidateIn = CandidateIn
    { candID        :: SegmentID
    , candBaseScore :: ScoreFix
    , candBitmask   :: Vec BitmaskWords (BitVector 64)
    , candPosition  :: Position
    , candTokens    :: Vec CandTokensSize (Maybe Token)
    , candValid     :: Bool
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

data RankedCandidate = RankedCandidate
    { rankedID    :: SegmentID
    , rankedScore :: ScoreFix
    , rankedPos   :: Position
    , rankedValid :: Bool
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

data HardwareCommand
    = CmdIdle
    | CmdInitQuery (Unsigned 64) (Vec BitmaskWords (BitVector 64)) Position (Vec CandTokensSize (Maybe Token))
    | CmdIngestQueryToken Token
    | CmdIngestToken TokenStreamIn
    | CmdScoreCandidate CandidateIn
    | CmdFlushTopK
    deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

data HardwareResponse = HardwareResponse
    { respTopK  :: RankedCandidate
    , respReady :: Bool
    , respDone  :: Bool
    , respBusy  :: Bool
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

hardwareHashStep :: Unsigned 64 -> Token -> Unsigned 64 -> Unsigned 64
hardwareHashStep seed tok laneSeed =
    let t64 = resize (unToken tok) :: Unsigned 64
        mixed = (t64 `xor` seed) `xor` laneSeed
        step1 = mixed * 0xbf58476d1ce4e5b9
        step2 = step1 `xor` (step1 .>>. 30)
        step3 = step2 * 0x94d049bb133111eb
    in step3 `xor` (step3 .>>. 31)

data MinHashState = MinHashState
    { mhsMins     :: Vec NumHashLanes (Unsigned 64)
    , mhsTokenCnt :: Unsigned 64
    , mhsSeed     :: Unsigned 64
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

initMinHashState :: Unsigned 64 -> MinHashState
initMinHashState seed = MinHashState
    { mhsMins     = repeat maxBound
    , mhsTokenCnt = 0
    , mhsSeed     = seed
    }

updateMinHash :: MinHashState -> Token -> MinHashState
updateMinHash st tok =
    let laneIndices = iterateI (+1) (0 :: Unsigned 64)
        computeLane idx currentMin =
            let laneSeedA = mhsSeed st + (idx * hashMultA)
                laneSeedB = mhsSeed st + ((idx + 1) * hashMultB)
                h = hardwareHashStep (mhsSeed st) tok laneSeedA `xor` laneSeedB
            in min currentMin h
        newMins = zipWith computeLane laneIndices (mhsMins st)
        newCount = if mhsTokenCnt st == maxBound then maxBound else mhsTokenCnt st + 1
    in st { mhsMins = newMins, mhsTokenCnt = newCount }

extractBitmask :: MinHashState -> Vec BitmaskWords (BitVector 64)
extractBitmask st =
    let bits = map (\m -> if testBit m 0 then (1 :: Bit) else 0) (mhsMins st)
        word0Bits = take (SNat :: SNat 64) bits
        word1Bits = drop (SNat :: SNat 64) bits
        w0 = pack word0Bits
        w1 = pack word1Bits
    in w0 :> w1 :> Nil

popCount64 :: BitVector 64 -> Unsigned 16
popCount64 bv =
    let u = unpack bv :: Vec 64 Bit
        addBit acc b = acc + (if b == 1 then 1 else 0)
    in foldl addBit 0 u

computeJaccardHW
    :: Vec BitmaskWords (BitVector 64)
    -> Vec BitmaskWords (BitVector 64)
    -> Unsigned 8
    -> ScoreFix
computeJaccardHW maskA maskB validBits =
    if validBits == 0 then ScoreFix 0.0
    else
        let validBitsClamped = min 128 (resize validBits :: Unsigned 16)
            v0 = min 64 validBitsClamped
            maskWord0 = if v0 == 64 then maxBound else (1 .<<. v0) - 1
            agree0 = complement (head maskA `xor` head maskB) .&. maskWord0
            matches0 = popCount64 agree0

            v1 = if validBitsClamped > 64 then validBitsClamped - 64 else 0
            maskWord1 = if v1 == 64 then maxBound else if v1 == 0 then 0 else (1 .<<. v1) - 1
            agree1 = complement (last maskA `xor` last maskB) .&. maskWord1
            matches1 = popCount64 agree1

            totalMatches = matches0 + matches1
            ratio = (fromIntegral totalMatches :: Fix16) / (fromIntegral validBitsClamped :: Fix16)
            estimate = (2.0 * ratio) - 1.0
            clamped = max 0.0 (min 1.0 estimate)
        in ScoreFix clamped

data ProximityState = ProximityState
    { psTotalDist :: Unsigned 64
    , psAnchorCnt :: Unsigned 64
    , psTokenCnt  :: Unsigned 64
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

initProximityState :: ProximityState
initProximityState = ProximityState 0 0 0

updateProximity :: ProximityState -> TokenStreamIn -> ProximityState
updateProximity st tin =
    if not (tsValid tin) then st
    else
        let currentPos = psTokenCnt st
            anchorPos = unPosition (tsAnchorPos tin)
            rawDist = if currentPos >= anchorPos then currentPos - anchorPos else anchorPos - currentPos
            clampedDist = min rawDist 0xFFFFFFFF
            newTotal = if tsHasAnchor tin
                       then (let sumVal = psTotalDist st + clampedDist
                             in if sumVal < psTotalDist st then maxBound else sumVal)
                       else psTotalDist st
            newAnchors = if tsHasAnchor tin
                         then (if psAnchorCnt st == maxBound then maxBound else psAnchorCnt st + 1)
                         else psAnchorCnt st
            newTokenCnt = if psTokenCnt st == maxBound then maxBound else psTokenCnt st + 1
        in ProximityState
            { psTotalDist = newTotal
            , psAnchorCnt = newAnchors
            , psTokenCnt  = newTokenCnt
            }

calculateProximityScore :: ProximityState -> ScoreFix
calculateProximityScore st =
    if psAnchorCnt st == 0 || psTokenCnt st == 0 then ScoreFix 0.0
    else
        let aCnt = psAnchorCnt st
            tCnt = psTokenCnt st
            denomProduct = aCnt * tCnt
            denom = if aCnt /= 0 && denomProduct `div` aCnt /= tCnt then maxBound else denomProduct
            denomSafe = max 1 denom
            totDist = psTotalDist st
            distRatio = if totDist >= denomSafe then (1.0 :: Fix16)
                        else (fromIntegral totDist :: Fix16) / (fromIntegral denomSafe :: Fix16)
            prox = max 0.0 (min 1.0 (1.0 - distRatio))
        in ScoreFix prox

computeCandidateDiversityAndOverlap
    :: TokenSet
    -> Vec CandTokensSize (Maybe Token)
    -> (ScoreFix, ScoreFix)
computeCandidateDiversityAndOverlap qSet cTokens =
    let processToken (cSet, ovCount, totalToks) maybeTok = case maybeTok of
            Nothing -> (cSet, ovCount, totalToks)
            Just tok ->
                let (newCSet, isNew) = insertToken cSet tok
                    newOv = if isNew && containsToken qSet tok then ovCount + 1 else ovCount
                in (newCSet, newOv, totalToks + (1 :: Unsigned 8))
        (finalCSet, overlapCnt, totalCount) =
            foldl processToken (initTokenSet, 0 :: Unsigned 8, 0 :: Unsigned 8) cTokens
        cUnique = resize (tsCount finalCSet) :: Unsigned 8
        qUnique = resize (tsCount qSet) :: Unsigned 8
        maxUnique = max qUnique cUnique
        divScore = if totalCount == 0 then (0.0 :: Fix16)
                   else min 1.0 ((fromIntegral cUnique :: Fix16) / (fromIntegral totalCount :: Fix16))
        ovScore = if maxUnique == 0 then (0.0 :: Fix16)
                  else min 1.0 ((fromIntegral overlapCnt :: Fix16) / (fromIntegral maxUnique :: Fix16))
    in (ScoreFix divScore, ScoreFix ovScore)

fuseScores
    :: ScoreFix
    -> ScoreFix
    -> ScoreFix
    -> ScoreFix
    -> ScoreFix
    -> ScoreFix
fuseScores (ScoreFix base) (ScoreFix overlap) (ScoreFix jaccard) (ScoreFix prox) (ScoreFix divScore) =
    let rawBase = max 0.0 (min 100.0 base)
        scaledBase = rawBase * fixMaxRawScoreInv
        partBase = scaledBase * fixWeightBase
        partOverlap = (max 0.0 (min 1.0 overlap)) * fixWeightOverlap
        partJaccard = (max 0.0 (min 1.0 jaccard)) * fixWeightJaccard
        partProx = (max 0.0 (min 1.0 prox)) * fixWeightProximity
        partDiv = (max 0.0 (min 1.0 divScore)) * fixWeightDiversity
        combined = partBase + partOverlap + partJaccard + partProx + partDiv
        finalClamped = max 0.0 (min 1.0 combined)
    in ScoreFix finalClamped

data SystolicCell = SystolicCell
    { cellItem :: RankedCandidate
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

initSystolicCell :: SystolicCell
initSystolicCell = SystolicCell (RankedCandidate (SegmentID 0) (ScoreFix (-1.0)) (Position 0) False)

stepSystolicArray
    :: Vec TopKDepth SystolicCell
    -> RankedCandidate
    -> (Vec TopKDepth SystolicCell, RankedCandidate)
stepSystolicArray cells incoming =
    let insertStep carry cell =
            let item = cellItem cell
                carryBetter = rankedValid carry && (not (rankedValid item) || rankedScore carry > rankedScore item)
                newCellItem = if carryBetter then carry else item
                newCarry    = if carryBetter then item else carry
            in (newCarry, SystolicCell newCellItem)
        (kickedOut, newCells) = mapAccumL insertStep incoming cells
    in (newCells, kickedOut)

data CoreFSM
    = StateIdle
    | StateStreaming
    | StateFlushing (Index TopKDepth)
    deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

data CoreState = CoreState
    { csFSM           :: CoreFSM
    , csQuerySeed     :: Unsigned 64
    , csQueryBitmask  :: Vec BitmaskWords (BitVector 64)
    , csQueryPosition :: Position
    , csQueryTokens   :: TokenSet
    , csCandTokens    :: TokenSet
    , csOverlapCount  :: Unsigned 8
    , csMinHash       :: MinHashState
    , csProximity     :: ProximityState
    , csSystolic      :: Vec TopKDepth SystolicCell
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

initialCoreState :: CoreState
initialCoreState = CoreState
    { csFSM           = StateIdle
    , csQuerySeed     = 0
    , csQueryBitmask  = repeat 0
    , csQueryPosition = Position 0
    , csQueryTokens   = initTokenSet
    , csCandTokens    = initTokenSet
    , csOverlapCount  = 0
    , csMinHash       = initMinHashState 0
    , csProximity     = initProximityState
    , csSystolic      = repeat initSystolicCell
    }

rankerEngineT
    :: CoreState
    -> HardwareCommand
    -> (CoreState, HardwareResponse)
rankerEngineT st cmd = case csFSM st of
    StateIdle -> case cmd of
        CmdInitQuery qSeed qMask qPos initialTokens ->
            let insertInitialTokens accSet maybeTok = case maybeTok of
                    Nothing -> accSet
                    Just tok -> fst (insertToken accSet tok)
                populatedQueryTokens = foldl insertInitialTokens initTokenSet initialTokens
                nextSt = st
                    { csFSM           = StateIdle
                    , csQuerySeed     = qSeed
                    , csQueryBitmask  = qMask
                    , csQueryPosition = qPos
                    , csQueryTokens   = populatedQueryTokens
                    , csCandTokens    = initTokenSet
                    , csOverlapCount  = 0
                    , csMinHash       = initMinHashState qSeed
                    , csProximity     = initProximityState
                    , csSystolic      = repeat initSystolicCell
                    }
                topElem = cellItem (head (csSystolic nextSt))
                resp = HardwareResponse topElem True False False
            in (nextSt, resp)

        CmdIngestQueryToken qTok ->
            let (nextQSet, _) = insertToken (csQueryTokens st) qTok
                nextSt = st { csQueryTokens = nextQSet }
                topElem = cellItem (head (csSystolic nextSt))
                resp = HardwareResponse topElem True False False
            in (nextSt, resp)

        CmdIngestToken tin ->
            if tsValid tin then
                let nextMH = updateMinHash (csMinHash st) (tsToken tin)
                    nextProx = updateProximity (csProximity st) tin
                    (nextCandSet, isNew) = insertToken (csCandTokens st) (tsToken tin)
                    nextOverlap = if isNew && containsToken (csQueryTokens st) (tsToken tin)
                                  then csOverlapCount st + 1
                                  else csOverlapCount st
                in if tsLast tin then
                    let localMask = extractBitmask nextMH
                        jaccard = computeJaccardHW (csQueryBitmask st) localMask 128
                        prox = calculateProximityScore nextProx
                        cUnique = resize (tsCount nextCandSet) :: Unsigned 8
                        qUnique = resize (tsCount (csQueryTokens st)) :: Unsigned 8
                        maxUnique = max qUnique cUnique
                        totTokens = resize (min 255 (psTokenCnt nextProx)) :: Unsigned 8
                        divScore = if totTokens == 0 then ScoreFix 0.0
                                   else ScoreFix (min 1.0 ((fromIntegral cUnique :: Fix16) / (fromIntegral totTokens :: Fix16)))
                        ovScore = if maxUnique == 0 then ScoreFix 0.0
                                  else ScoreFix (min 1.0 ((fromIntegral nextOverlap :: Fix16) / (fromIntegral maxUnique :: Fix16)))
                        fused = fuseScores (tsBaseScore tin) ovScore jaccard prox divScore
                        candRanked = RankedCandidate (tsSegmentID tin) fused (tsPosition tin) True
                        (nextSystolic, _) = stepSystolicArray (csSystolic st) candRanked
                        nextSt = st
                            { csFSM          = StateIdle
                            , csCandTokens   = initTokenSet
                            , csOverlapCount = 0
                            , csMinHash      = initMinHashState (csQuerySeed st)
                            , csProximity    = initProximityState
                            , csSystolic     = nextSystolic
                            }
                        topElem = cellItem (head nextSystolic)
                        resp = HardwareResponse topElem True False False
                    in (nextSt, resp)
                else
                    let nextSt = st
                            { csFSM          = StateStreaming
                            , csCandTokens   = nextCandSet
                            , csOverlapCount = nextOverlap
                            , csMinHash      = nextMH
                            , csProximity    = nextProx
                            }
                        topElem = cellItem (head (csSystolic st))
                        resp = HardwareResponse topElem True False True
                    in (nextSt, resp)
            else
                let topElem = cellItem (head (csSystolic st))
                    resp = HardwareResponse topElem True False False
                in (st, resp)

        CmdScoreCandidate cand ->
            if candValid cand then
                let jaccard = computeJaccardHW (csQueryBitmask st) (candBitmask cand) 128
                    qPos = unPosition (csQueryPosition st)
                    cPos = unPosition (candPosition cand)
                    pDist = if cPos >= qPos then cPos - qPos else qPos - cPos
                    pClamped = min pDist 1000
                    pScore = 1.0 - ((fromIntegral pClamped :: Fix16) / 1000.0)
                    prox = ScoreFix (max 0.0 (min 1.0 pScore))
                    (divScore, ovScore) = computeCandidateDiversityAndOverlap (csQueryTokens st) (candTokens cand)
                    fused = fuseScores (candBaseScore cand) ovScore jaccard prox divScore
                    candRanked = RankedCandidate (candID cand) fused (candPosition cand) True
                    (nextSystolic, _) = stepSystolicArray (csSystolic st) candRanked
                    nextSt = st { csSystolic = nextSystolic }
                    topElem = cellItem (head nextSystolic)
                    resp = HardwareResponse topElem True False False
                in (nextSt, resp)
            else
                let topElem = cellItem (head (csSystolic st))
                    resp = HardwareResponse topElem True False False
                in (st, resp)

        CmdFlushTopK ->
            let currentItem = cellItem (head (csSystolic st))
                isDone = (maxBound :: Index TopKDepth) == minBound
                nextFSM = if isDone then StateIdle else StateFlushing 1
                nextSt = st { csFSM = nextFSM }
                resp = HardwareResponse currentItem False isDone (not isDone)
            in (nextSt, resp)

        CmdIdle ->
            let topElem = cellItem (head (csSystolic st))
                resp = HardwareResponse topElem True False False
            in (st, resp)

    StateStreaming -> case cmd of
        CmdIngestToken tin ->
            if tsValid tin then
                let nextMH = updateMinHash (csMinHash st) (tsToken tin)
                    nextProx = updateProximity (csProximity st) tin
                    (nextCandSet, isNew) = insertToken (csCandTokens st) (tsToken tin)
                    nextOverlap = if isNew && containsToken (csQueryTokens st) (tsToken tin)
                                  then csOverlapCount st + 1
                                  else csOverlapCount st
                in if tsLast tin then
                    let localMask = extractBitmask nextMH
                        jaccard = computeJaccardHW (csQueryBitmask st) localMask 128
                        prox = calculateProximityScore nextProx
                        cUnique = resize (tsCount nextCandSet) :: Unsigned 8
                        qUnique = resize (tsCount (csQueryTokens st)) :: Unsigned 8
                        maxUnique = max qUnique cUnique
                        totTokens = resize (min 255 (psTokenCnt nextProx)) :: Unsigned 8
                        divScore = if totTokens == 0 then ScoreFix 0.0
                                   else ScoreFix (min 1.0 ((fromIntegral cUnique :: Fix16) / (fromIntegral totTokens :: Fix16)))
                        ovScore = if maxUnique == 0 then ScoreFix 0.0
                                  else ScoreFix (min 1.0 ((fromIntegral nextOverlap :: Fix16) / (fromIntegral maxUnique :: Fix16)))
                        fused = fuseScores (tsBaseScore tin) ovScore jaccard prox divScore
                        candRanked = RankedCandidate (tsSegmentID tin) fused (tsPosition tin) True
                        (nextSystolic, _) = stepSystolicArray (csSystolic st) candRanked
                        nextSt = st
                            { csFSM          = StateIdle
                            , csCandTokens   = initTokenSet
                            , csOverlapCount = 0
                            , csMinHash      = initMinHashState (csQuerySeed st)
                            , csProximity    = initProximityState
                            , csSystolic     = nextSystolic
                            }
                        topElem = cellItem (head nextSystolic)
                        resp = HardwareResponse topElem True False False
                    in (nextSt, resp)
                else
                    let nextSt = st
                            { csFSM          = StateStreaming
                            , csCandTokens   = nextCandSet
                            , csOverlapCount = nextOverlap
                            , csMinHash      = nextMH
                            , csProximity    = nextProx
                            }
                        topElem = cellItem (head (csSystolic st))
                        resp = HardwareResponse topElem True False True
                    in (nextSt, resp)
            else
                let topElem = cellItem (head (csSystolic st))
                    resp = HardwareResponse topElem True False True
                in (st, resp)

        _ ->
            let topElem = cellItem (head (csSystolic st))
                resp = HardwareResponse topElem False False True
            in (st, resp)

    StateFlushing idx ->
        let currentItem = cellItem (csSystolic st !! idx)
            isDone = idx == maxBound
            nextFSM = if isDone then StateIdle else StateFlushing (idx + 1)
            nextSt = st { csFSM = nextFSM }
            resp = HardwareResponse currentItem (if isDone then True else False) isDone (if isDone then False else True)
        in (nextSt, resp)

rankerCore
    :: HiddenClockResetEnable dom
    => Signal dom HardwareCommand
    -> Signal dom HardwareResponse
rankerCore = mealy rankerEngineT initialCoreState

{-# NOINLINE topEntity #-}
topEntity
    :: Clock System
    -> Reset System
    -> Enable System
    -> Signal System HardwareCommand
    -> Signal System HardwareResponse
topEntity = exposeClockResetEnable rankerCore