{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module SSISearch
    ( HashKey64(..)
    , NodeAddr32(..)
    , SearchOp(..)
    , NodeType(..)
    , SSISegment(..)
    , SearchRequest(..)
    , SearchResult(..)
    , TreeNode(..)
    , SearchStatus(..)
    , SearchState(..)
    , MaxSearchDepthConfig
    , maxSearchDepthVal
    , nullAddr
    , emptySegment
    , bucketWidthVal
    , bucketCountVal
    , tensorWidthVal
    , mkSegment
    , mkSearchRequest
    , mkPositionSearchRequest
    , mkSimilaritySearchRequest
    , mkUpdateScoreRequest
    , mkSearchResult
    , mkTreeNode
    , mkRootNode
    , mkLeafNode
    , mkCollisionNode
    , mixHash
    , minHashSeedA
    , minHashSeedB
    , minHashLaneHash
    , minHashLaneForTokens
    , computeMinHashSignature
    , hashTokens
    , hashTokensLen
    , computeAnchorHash
    , bucketIndex
    , segmentFullHash
    , computeLeafHash
    , computeBranchHash
    , countBits64
    , isqrt32
    , signatureSimilarity
    , computeSimilarity
    , computeFusedSimilarity
    , low32
    , high32
    , joinU64
    , findNextBucket
    , findFirstBucket
    , ssiSearch
    , ssiSearchT
    , checkNode
    , topEntity
    ) where

import Clash.Prelude
import GHC.TypeLits

newtype HashKey64 = HashKey64 { unHashKey64 :: Unsigned 64 }
    deriving stock (Generic, Eq, Ord, Show)
    deriving anyclass (NFDataX, ShowX, BitPack, Default)

newtype NodeAddr32 = NodeAddr32 { unNodeAddr32 :: Unsigned 32 }
    deriving stock (Generic, Eq, Ord, Show)
    deriving anyclass (NFDataX, ShowX, BitPack, Default)

data SearchOp
    = OpLookupPosition
    | OpLookupExactKey
    | OpSearchSimilarity
    | OpUpdateScore
    deriving stock (Generic, Eq, Show)
    deriving anyclass (NFDataX, ShowX, BitPack, Default)

data NodeType
    = NodeRoot
    | NodeLeaf
    | NodeCollision
    deriving stock (Generic, Eq, Show)
    deriving anyclass (NFDataX, ShowX, BitPack, Default)

data SSISegment = SSISegment
    { segPosition   :: Unsigned 64
    , segScore      :: Unsigned 32
    , segAnchorHash :: Unsigned 64
    , segSignature  :: Unsigned 64
    , segTokenHash  :: Unsigned 64
    , segTokenCount :: Unsigned 8
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack, Default)

data SearchRequest = SearchRequest
    { searchOp       :: SearchOp
    , searchKey      :: HashKey64
    , searchPosition :: Unsigned 64
    , querySignature :: Unsigned 64
    , queryScore     :: Unsigned 32
    , rootAddr       :: NodeAddr32
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack, Default)

data SearchResult = SearchResult
    { foundAddr        :: NodeAddr32
    , found            :: Bool
    , depth            :: Unsigned 8
    , resultSegment    :: SSISegment
    , resultSimilarity :: Unsigned 32
    , resultHash       :: HashKey64
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack, Default)

data TreeNode = TreeNode
    { nodeKey       :: HashKey64
    , nodeType      :: NodeType
    , rootBuckets   :: Vec 64 NodeAddr32
    , segmentData   :: SSISegment
    , collisionNext :: NodeAddr32
    , leftChild     :: NodeAddr32
    , rightChild    :: NodeAddr32
    , isValid       :: Bool
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack, Default)

data SearchStatus
    = Searching
    | Found (Index 64, NodeAddr32)
    deriving stock (Generic, Eq, Show)
    deriving anyclass (NFDataX, ShowX, BitPack)

data SearchState
    = Idle
    | FetchingRoot SearchRequest NodeAddr32 (Unsigned 8)
    | FetchingLeaf SearchRequest NodeAddr32 (Unsigned 8)
    | FetchingCollision SearchRequest NodeAddr32 (Unsigned 8)
    | ScanningBucket SearchRequest (Vec 64 NodeAddr32) (Index 64) (Unsigned 8) NodeAddr32 SSISegment (Unsigned 32)
    | ScanningCollision SearchRequest (Vec 64 NodeAddr32) (Index 64) (Unsigned 8) NodeAddr32 SSISegment (Unsigned 32) NodeAddr32
    deriving stock (Generic, Eq, Show)
    deriving anyclass (NFDataX, ShowX, BitPack)

type MaxSearchDepthConfig = 64 :: Nat

maxSearchDepthVal :: Unsigned 8
maxSearchDepthVal = snatToNum (SNat :: SNat MaxSearchDepthConfig)

nullAddr :: NodeAddr32
nullAddr = NodeAddr32 0

emptySegment :: SSISegment
emptySegment = SSISegment 0 0 0 0 0 0

bucketWidthVal :: Unsigned 8
bucketWidthVal = 6

bucketCountVal :: Unsigned 8
bucketCountVal = 64

tensorWidthVal :: Unsigned 8
tensorWidthVal = 134

mkSegment :: Unsigned 64 -> Unsigned 32 -> Unsigned 64 -> Unsigned 64 -> Unsigned 64 -> Unsigned 8 -> SSISegment
mkSegment pos sc anc sig th count = SSISegment pos sc anc sig th count

mkSearchRequest :: Unsigned 64 -> Unsigned 32 -> SearchRequest
mkSearchRequest k r = SearchRequest OpLookupExactKey (HashKey64 k) (resize k) 0 0 (NodeAddr32 r)

mkPositionSearchRequest :: Unsigned 64 -> Unsigned 32 -> SearchRequest
mkPositionSearchRequest pos r = SearchRequest OpLookupPosition (HashKey64 (mixHash 0 pos)) pos 0 0 (NodeAddr32 r)

mkSimilaritySearchRequest :: Unsigned 64 -> Unsigned 64 -> Unsigned 32 -> SearchRequest
mkSimilaritySearchRequest qHash qSig r = SearchRequest OpSearchSimilarity (HashKey64 qHash) 0 qSig 0 (NodeAddr32 r)

mkUpdateScoreRequest :: Unsigned 64 -> Unsigned 32 -> Unsigned 32 -> SearchRequest
mkUpdateScoreRequest pos newScore r = SearchRequest OpUpdateScore (HashKey64 (mixHash 0 pos)) pos 0 newScore (NodeAddr32 r)

mkSearchResult :: NodeAddr32 -> Bool -> Unsigned 8 -> SearchResult
mkSearchResult addr f d = SearchResult addr f d emptySegment 0 (HashKey64 0)

mkTreeNode :: Unsigned 64 -> Unsigned 32 -> Unsigned 32 -> Bool -> TreeNode
mkTreeNode k l r v = TreeNode (HashKey64 k) NodeLeaf (repeat nullAddr) emptySegment (NodeAddr32 r) (NodeAddr32 l) (NodeAddr32 r) v

mkRootNode :: Unsigned 64 -> Vec 64 NodeAddr32 -> TreeNode
mkRootNode h buckets = TreeNode (HashKey64 h) NodeRoot buckets emptySegment nullAddr nullAddr nullAddr True

mkLeafNode :: Unsigned 64 -> SSISegment -> NodeAddr32 -> TreeNode
mkLeafNode h seg next = TreeNode (HashKey64 h) NodeLeaf (repeat nullAddr) seg next nullAddr nullAddr True

mkCollisionNode :: SSISegment -> NodeAddr32 -> TreeNode
mkCollisionNode seg next = TreeNode (HashKey64 (segmentFullHash seg)) NodeCollision (repeat nullAddr) seg next nullAddr nullAddr True

mixHash :: Unsigned 64 -> Unsigned 64 -> Unsigned 64
mixHash state value = (state * 0x9E3779B185EBCA87) + value + 0x517CC1B727220A95

minHashSeedA :: Unsigned 64 -> Unsigned 64
minHashSeedA lane = 0x9E3779B185EBCA87 + (lane * 0xC2B2AE3D27D4EB4F) + 0x165667B19E3779F9

minHashSeedB :: Unsigned 64 -> Unsigned 64
minHashSeedB lane = 0x517CC1B727220A95 + (lane * 0xD6E8FEB86659FD93) + 0x2545F4914F6CDD1D

minHashLaneHash :: Unsigned 32 -> Unsigned 64 -> Unsigned 64
minHashLaneHash token lane =
    let h0 = (resize token * minHashSeedA lane) + minHashSeedB lane
        h1 = (h0 `xor` (h0 `shiftR` 30)) * 0xbf58476d1ce4e5b9
        h2 = (h1 `xor` (h1 `shiftR` 27)) * 0x94d049bb133111eb
        h3 = h2 `xor` (h2 `shiftR` 31)
    in h3

minHashLaneForTokens :: KnownNat n => Vec n (Unsigned 32) -> Index 64 -> Unsigned 64
minHashLaneForTokens tokens laneIdx =
    let lane = fromIntegral laneIdx :: Unsigned 64
        laneHashes = map (\tok -> minHashLaneHash tok lane) tokens
    in foldl min maxBound laneHashes

computeMinHashSignature :: KnownNat n => Vec n (Unsigned 32) -> Unsigned 64
computeMinHashSignature tokens =
    let laneIndices = indicesI :: Vec 64 (Index 64)
        processLane sig laneIdx =
            let minVal = minHashLaneForTokens tokens laneIdx
            in if (minVal .&. 1) /= 0
               then sig .|. (1 `shiftL` fromIntegral laneIdx)
               else sig
    in foldl processLane 0 laneIndices

hashTokens :: forall n. KnownNat n => Vec n (Unsigned 32) -> Unsigned 64
hashTokens tokens =
    let len = snatToNum (SNat :: SNat n) :: Unsigned 64
        init0 = mixHash 0 len
    in foldl mixHash init0 (map resize tokens)

hashTokensLen :: forall n. KnownNat n => Vec n (Unsigned 32) -> Unsigned 8 -> Unsigned 64
hashTokensLen tokens actualLen =
    let init0 = mixHash 0 (resize actualLen)
        folder (st, idx) tok =
            if idx < actualLen
            then (mixHash st (resize tok), idx + 1)
            else (st, idx + 1)
    in fst (foldl folder (init0, 0) tokens)

computeAnchorHash :: forall n. KnownNat n => Vec n (Unsigned 32) -> Unsigned 64 -> Unsigned 64
computeAnchorHash tokens pos =
    let len = snatToNum (SNat :: SNat n) :: Unsigned 64
        init0 = mixHash pos len
    in foldl mixHash init0 (map resize tokens)

bucketIndex :: Unsigned 64 -> Index 64
bucketIndex pos =
    let h0 = pos * 0x9E3779B185EBCA87
        h1 = (h0 `xor` (h0 `shiftR` 30)) * 0xbf58476d1ce4e5b9
        h2 = (h1 `xor` (h1 `shiftR` 27)) * 0x94d049bb133111eb
        h3 = h2 `xor` (h2 `shiftR` 31)
    in fromIntegral (h3 .&. 0x3F)

segmentFullHash :: SSISegment -> Unsigned 64
segmentFullHash seg =
    let s0 = mixHash 0 (segPosition seg)
        s1 = mixHash s0 (resize (segScore seg))
        s2 = mixHash s1 (segAnchorHash seg)
        s3 = mixHash s2 (segSignature seg)
        s4 = mixHash s3 (resize (segTokenCount seg))
        s5 = mixHash s4 (segTokenHash seg)
    in s5

computeLeafHash :: TreeNode -> Unsigned 64
computeLeafHash node =
    let baseHash = if isValid node then segmentFullHash (segmentData node) else 0
        chainHash = unHashKey64 (nodeKey node)
    in baseHash + chainHash

computeBranchHash :: Vec 64 (Unsigned 64) -> Unsigned 64
computeBranchHash hashes = foldl (+) 0 hashes

countBits64 :: Unsigned 64 -> Unsigned 8
countBits64 v = foldl (+) 0 (map (\b -> if b == high then 1 else 0) (bv2v (pack v)))

isqrt32 :: Unsigned 32 -> Unsigned 32
isqrt32 val = foldl step 0 (reverse (indicesI :: Vec 16 (Index 16)))
  where
    step acc i =
        let bit = 1 `shiftL` fromIntegral i
            cand = acc .|. bit
        in if cand * cand <= val then cand else acc

signatureSimilarity :: Unsigned 64 -> Unsigned 64 -> Unsigned 32
signatureSimilarity qSig sSig =
    let mismatch = countBits64 (qSig `xor` sSig)
    in if mismatch >= 32
       then 0
       else resize (32 - mismatch) * 2048

computeSimilarity :: Unsigned 64 -> Unsigned 64 -> Unsigned 32
computeSimilarity h1 h2 =
    let pc1 = countBits64 h1
        pc2 = countBits64 h2
        inter = countBits64 (h1 .&. h2)
    in if pc1 == 0 && pc2 == 0
       then 65536
       else if pc1 == 0 || pc2 == 0
            then 0
            else
                let prod = resize pc1 * resize pc2 :: Unsigned 32
                    root = isqrt32 prod
                in if root == 0
                   then 0
                   else (resize inter * 65536) `div` root

computeFusedSimilarity :: Unsigned 64 -> Unsigned 64 -> Unsigned 64 -> Unsigned 64 -> Unsigned 32
computeFusedSimilarity qHash qSig sHash sSig =
    let cosSim = computeSimilarity qHash sHash
        jacSim = signatureSimilarity qSig sSig
    in (cosSim + jacSim) `shiftR` 1

low32 :: Unsigned 64 -> Unsigned 32
low32 v = resize (v .&. 0xFFFFFFFF)

high32 :: Unsigned 64 -> Unsigned 32
high32 v = resize (v `shiftR` 32)

joinU64 :: Unsigned 32 -> Unsigned 32 -> Unsigned 64
joinU64 lo hi = (resize hi `shiftL` 32) .|. resize lo

findNextBucket :: Vec 64 NodeAddr32 -> Index 64 -> Maybe (Index 64, NodeAddr32)
findNextBucket buckets startIdx =
    let searchIndices = indicesI :: Vec 64 (Index 64)
        step (Found res) _ = Found res
        step Searching idx =
            if idx > startIdx && (buckets !! idx) /= nullAddr
            then Found (idx, buckets !! idx)
            else Searching
    in case foldl step Searching searchIndices of
        Found (i, a) -> Just (i, a)
        Searching    -> Nothing

findFirstBucket :: Vec 64 NodeAddr32 -> Maybe (Index 64, NodeAddr32)
findFirstBucket buckets =
    if (buckets !! (0 :: Index 64)) /= nullAddr
    then Just (0, buckets !! (0 :: Index 64))
    else findNextBucket buckets 0

ssiSearch
    :: HiddenClockResetEnable dom
    => Signal dom (Maybe SearchRequest)
    -> Signal dom (Maybe TreeNode)
    -> (Signal dom (Maybe NodeAddr32), Signal dom (Maybe SearchResult))
ssiSearch reqIn nodeIn = (memReq, resultOut)
  where
    (memReq, resultOut) = unbundle $ mealy ssiSearchT Idle (bundle (reqIn, nodeIn))

ssiSearchT
    :: SearchState
    -> (Maybe SearchRequest, Maybe TreeNode)
    -> (SearchState, (Maybe NodeAddr32, Maybe SearchResult))
ssiSearchT Idle (Just req, _)
    | rootAddr req == nullAddr =
        let notFound = SearchResult nullAddr False 0 emptySegment 0 (HashKey64 0)
        in (Idle, (Nothing, Just notFound))
    | otherwise =
        (FetchingRoot req (rootAddr req) 1, (Just (rootAddr req), Nothing))
ssiSearchT Idle _ = (Idle, (Nothing, Nothing))

ssiSearchT (FetchingRoot req addr currentDepth) (_, Just node)
    | not (isValid node) = (Idle, (Nothing, Just notFound))
    | nodeType node == NodeRoot = case searchOp req of
        OpLookupPosition ->
            let bIdx = bucketIndex (searchPosition req)
                targetAddr = rootBuckets node !! bIdx
            in if targetAddr == nullAddr
               then (Idle, (Nothing, Just notFound))
               else if currentDepth >= maxSearchDepthVal
               then (Idle, (Nothing, Just depthExceeded))
               else (FetchingLeaf req targetAddr (currentDepth + 1), (Just targetAddr, Nothing))
        OpUpdateScore ->
            let bIdx = bucketIndex (searchPosition req)
                targetAddr = rootBuckets node !! bIdx
            in if targetAddr == nullAddr
               then (Idle, (Nothing, Just notFound))
               else if currentDepth >= maxSearchDepthVal
               then (Idle, (Nothing, Just depthExceeded))
               else (FetchingLeaf req targetAddr (currentDepth + 1), (Just targetAddr, Nothing))
        OpSearchSimilarity ->
            case findFirstBucket (rootBuckets node) of
                Just (firstIdx, firstAddr) ->
                    if currentDepth >= maxSearchDepthVal
                    then (Idle, (Nothing, Just depthExceeded))
                    else (ScanningBucket req (rootBuckets node) firstIdx (currentDepth + 1) nullAddr emptySegment 0, (Just firstAddr, Nothing))
                Nothing ->
                    (Idle, (Nothing, Just notFound))
        OpLookupExactKey ->
            let bIdx = bucketIndex (unHashKey64 (searchKey req))
                targetAddr = rootBuckets node !! bIdx
            in if targetAddr /= nullAddr
               then if currentDepth >= maxSearchDepthVal
                    then (Idle, (Nothing, Just depthExceeded))
                    else (FetchingLeaf req targetAddr (currentDepth + 1), (Just targetAddr, Nothing))
               else checkNode req addr currentDepth node
    | otherwise = checkNode req addr currentDepth node
  where
    notFound = SearchResult nullAddr False currentDepth emptySegment 0 (HashKey64 0)
    depthExceeded = SearchResult nullAddr False currentDepth emptySegment 0 (HashKey64 0)
ssiSearchT (FetchingRoot req addr currentDepth) _ =
    (FetchingRoot req addr currentDepth, (Nothing, Nothing))

ssiSearchT (FetchingLeaf req addr currentDepth) (_, Just node)
    | not (isValid node) = (Idle, (Nothing, Just notFound))
    | otherwise = checkNode req addr currentDepth node
  where
    notFound = SearchResult nullAddr False currentDepth emptySegment 0 (HashKey64 0)
ssiSearchT (FetchingLeaf req addr currentDepth) _ =
    (FetchingLeaf req addr currentDepth, (Nothing, Nothing))

ssiSearchT (FetchingCollision req addr currentDepth) (_, Just node)
    | not (isValid node) = (Idle, (Nothing, Just notFound))
    | otherwise = checkNode req addr currentDepth node
  where
    notFound = SearchResult nullAddr False currentDepth emptySegment 0 (HashKey64 0)
ssiSearchT (FetchingCollision req addr currentDepth) _ =
    (FetchingCollision req addr currentDepth, (Nothing, Nothing))

ssiSearchT (ScanningBucket req buckets bIdx currentDepth bestAddr bestSeg bestScore) (_, Just node)
    | not (isValid node) =
        case findNextBucket buckets bIdx of
            Just (nextIdx, nextAddr) ->
                if currentDepth < maxSearchDepthVal
                then (ScanningBucket req buckets nextIdx (currentDepth + 1) bestAddr bestSeg bestScore, (Just nextAddr, Nothing))
                else (Idle, (Nothing, Just (SearchResult bestAddr (bestAddr /= nullAddr) currentDepth bestSeg bestScore (HashKey64 (segTokenHash bestSeg)))))
            Nothing ->
                let finalRes = SearchResult bestAddr (bestAddr /= nullAddr) currentDepth bestSeg bestScore (HashKey64 (segTokenHash bestSeg))
                in (Idle, (Nothing, Just finalRes))
    | otherwise =
        let sim = computeFusedSimilarity (unHashKey64 (searchKey req)) (querySignature req) (segTokenHash (segmentData node)) (segSignature (segmentData node))
            (nBestAddr, nBestSeg, nBestScore) = if sim >= bestScore then (buckets !! bIdx, segmentData node, sim) else (bestAddr, bestSeg, bestScore)
        in if collisionNext node /= nullAddr && currentDepth < maxSearchDepthVal
           then (ScanningCollision req buckets bIdx (currentDepth + 1) nBestAddr nBestSeg nBestScore (collisionNext node), (Just (collisionNext node), Nothing))
           else case findNextBucket buckets bIdx of
                Just (nextIdx, nextAddr) ->
                    if currentDepth < maxSearchDepthVal
                    then (ScanningBucket req buckets nextIdx (currentDepth + 1) nBestAddr nBestSeg nBestScore, (Just nextAddr, Nothing))
                    else (Idle, (Nothing, Just (SearchResult nBestAddr (nBestAddr /= nullAddr) currentDepth nBestSeg nBestScore (HashKey64 (segTokenHash nBestSeg)))))
                Nothing ->
                    let finalRes = SearchResult nBestAddr (nBestAddr /= nullAddr) currentDepth nBestSeg nBestScore (HashKey64 (segTokenHash nBestSeg))
                    in (Idle, (Nothing, Just finalRes))
ssiSearchT (ScanningBucket req buckets bIdx currentDepth bestAddr bestSeg bestScore) _ =
    (ScanningBucket req buckets bIdx currentDepth bestAddr bestSeg bestScore, (Nothing, Nothing))

ssiSearchT (ScanningCollision req buckets bIdx currentDepth bestAddr bestSeg bestScore colAddr) (_, Just node)
    | not (isValid node) =
        case findNextBucket buckets bIdx of
            Just (nextIdx, nextAddr) ->
                if currentDepth < maxSearchDepthVal
                then (ScanningBucket req buckets nextIdx (currentDepth + 1) bestAddr bestSeg bestScore, (Just nextAddr, Nothing))
                else (Idle, (Nothing, Just (SearchResult bestAddr (bestAddr /= nullAddr) currentDepth bestSeg bestScore (HashKey64 (segTokenHash bestSeg)))))
            Nothing ->
                let finalRes = SearchResult bestAddr (bestAddr /= nullAddr) currentDepth bestSeg bestScore (HashKey64 (segTokenHash bestSeg))
                in (Idle, (Nothing, Just finalRes))
    | otherwise =
        let sim = computeFusedSimilarity (unHashKey64 (searchKey req)) (querySignature req) (segTokenHash (segmentData node)) (segSignature (segmentData node))
            (nBestAddr, nBestSeg, nBestScore) = if sim >= bestScore then (colAddr, segmentData node, sim) else (bestAddr, bestSeg, bestScore)
        in if collisionNext node /= nullAddr && currentDepth < maxSearchDepthVal
           then (ScanningCollision req buckets bIdx (currentDepth + 1) nBestAddr nBestSeg nBestScore (collisionNext node), (Just (collisionNext node), Nothing))
           else case findNextBucket buckets bIdx of
                Just (nextIdx, nextAddr) ->
                    if currentDepth < maxSearchDepthVal
                    then (ScanningBucket req buckets nextIdx (currentDepth + 1) nBestAddr nBestSeg nBestScore, (Just nextAddr, Nothing))
                    else (Idle, (Nothing, Just (SearchResult nBestAddr (nBestAddr /= nullAddr) currentDepth nBestSeg nBestScore (HashKey64 (segTokenHash nBestSeg)))))
                Nothing ->
                    let finalRes = SearchResult nBestAddr (nBestAddr /= nullAddr) currentDepth nBestSeg nBestScore (HashKey64 (segTokenHash nBestSeg))
                    in (Idle, (Nothing, Just finalRes))
ssiSearchT (ScanningCollision req buckets bIdx currentDepth bestAddr bestSeg bestScore colAddr) _ =
    (ScanningCollision req buckets bIdx currentDepth bestAddr bestSeg bestScore colAddr, (Nothing, Nothing))

checkNode
    :: SearchRequest
    -> NodeAddr32
    -> Unsigned 8
    -> TreeNode
    -> (SearchState, (Maybe NodeAddr32, Maybe SearchResult))
checkNode req addr currentDepth node
    | not (isValid node) =
        (Idle, (Nothing, Just notFound))
    | searchOp req == OpLookupPosition =
        if segPosition (segmentData node) == searchPosition req
        then (Idle, (Nothing, Just (SearchResult addr True currentDepth (segmentData node) 65536 (nodeKey node))))
        else if collisionNext node /= nullAddr && currentDepth < maxSearchDepthVal
        then (FetchingCollision req (collisionNext node) (currentDepth + 1), (Just (collisionNext node), Nothing))
        else (Idle, (Nothing, Just (if currentDepth >= maxSearchDepthVal then depthExceeded else notFound)))
    | searchOp req == OpUpdateScore =
        if segPosition (segmentData node) == searchPosition req
        then let updatedSeg = (segmentData node) { segScore = queryScore req }
             in (Idle, (Nothing, Just (SearchResult addr True currentDepth updatedSeg 65536 (nodeKey node))))
        else if collisionNext node /= nullAddr && currentDepth < maxSearchDepthVal
        then (FetchingCollision req (collisionNext node) (currentDepth + 1), (Just (collisionNext node), Nothing))
        else (Idle, (Nothing, Just (if currentDepth >= maxSearchDepthVal then depthExceeded else notFound)))
    | searchOp req == OpSearchSimilarity =
        let sim = computeFusedSimilarity (unHashKey64 (searchKey req)) (querySignature req) (segTokenHash (segmentData node)) (segSignature (segmentData node))
        in (Idle, (Nothing, Just (SearchResult addr True currentDepth (segmentData node) sim (nodeKey node))))
    | searchKey req == nodeKey node || segTokenHash (segmentData node) == unHashKey64 (searchKey req) =
        (Idle, (Nothing, Just (SearchResult addr True currentDepth (segmentData node) 65536 (nodeKey node))))
    | otherwise = case compare (searchKey req) (nodeKey node) of
        LT -> if leftChild node /= nullAddr && currentDepth < maxSearchDepthVal
              then (FetchingLeaf req (leftChild node) (currentDepth + 1), (Just (leftChild node), Nothing))
              else if collisionNext node /= nullAddr && currentDepth < maxSearchDepthVal
              then (FetchingCollision req (collisionNext node) (currentDepth + 1), (Just (collisionNext node), Nothing))
              else (Idle, (Nothing, Just (if currentDepth >= maxSearchDepthVal then depthExceeded else notFound)))
        GT -> if rightChild node /= nullAddr && currentDepth < maxSearchDepthVal
              then (FetchingLeaf req (rightChild node) (currentDepth + 1), (Just (rightChild node), Nothing))
              else if collisionNext node /= nullAddr && currentDepth < maxSearchDepthVal
              then (FetchingCollision req (collisionNext node) (currentDepth + 1), (Just (collisionNext node), Nothing))
              else (Idle, (Nothing, Just (if currentDepth >= maxSearchDepthVal then depthExceeded else notFound)))
        EQ -> (Idle, (Nothing, Just (SearchResult addr True currentDepth (segmentData node) 65536 (nodeKey node))))
  where
    notFound = SearchResult nullAddr False currentDepth emptySegment 0 (HashKey64 0)
    depthExceeded = SearchResult nullAddr False currentDepth emptySegment 0 (HashKey64 0)

{-# NOINLINE topEntity #-}
{-# ANN topEntity
  (Synthesize
    { t_name   = "SSISearch"
    , t_inputs = [ PortName "clk"
                 , PortName "rst"
                 , PortName "en"
                 , PortName "reqIn"
                 , PortName "nodeIn"
                 ]
    , t_output = PortProduct "" [ PortName "memReq"
                                , PortName "resultOut"
                                ]
    }) #-}
topEntity
    :: Clock System
    -> Reset System
    -> Enable System
    -> Signal System (Maybe SearchRequest)
    -> Signal System (Maybe TreeNode)
    -> (Signal System (Maybe NodeAddr32), Signal System (Maybe SearchResult))
topEntity = exposeClockResetEnable ssiSearch