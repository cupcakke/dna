{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleContexts #-}

module MemoryArbiter
    ( Addr32(..)
    , Data64(..)
    , ClientID4(..)
    , NumClients
    , ServiceCycles
    , MemRequest(..)
    , MemResponse(..)
    , ArbiterState(..)
    , serviceCyclesVal
    , mkMemRequest
    , mkMemResponse
    , filterResp
    , arbiterT
    , initialArbiterState
    , grantForRequestMask
    , memoryArbiter
    , topEntity
    , testInput
    ) where

import Clash.Prelude

newtype Addr32 = Addr32 { unAddr32 :: Unsigned 32 }
    deriving stock (Generic, Eq, Ord, Show)
    deriving newtype (NFDataX, ShowX, BitPack)

newtype Data64 = Data64 { unData64 :: Unsigned 64 }
    deriving stock (Generic, Eq, Ord, Show)
    deriving newtype (NFDataX, ShowX, BitPack)

newtype ClientID4 = ClientID4 { unClientID4 :: Unsigned 4 }
    deriving stock (Generic, Eq, Ord, Show)
    deriving newtype (NFDataX, ShowX, BitPack)

type NumClients = 4
type ServiceCycles = 4

serviceCyclesVal :: Unsigned 8
serviceCyclesVal = snatToNum (SNat :: SNat ServiceCycles)

data MemRequest = MemRequest
    { reqAddr   :: Addr32
    , reqWrite  :: Bool
    , reqData   :: Data64
    , reqClient :: ClientID4
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

data MemResponse = MemResponse
    { respData   :: Data64
    , respClient :: ClientID4
    , respValid  :: Bool
    } deriving stock (Generic, Eq, Show)
      deriving anyclass (NFDataX, ShowX, BitPack)

data ArbiterState
    = ArbIdle
    | ArbServing ClientID4 (Unsigned 8)
    deriving stock (Generic, Eq, Show)
    deriving anyclass (NFDataX, ShowX)

initialArbiterState :: (ArbiterState, Unsigned 8)
initialArbiterState = (ArbIdle, 0)

mkMemRequest :: Unsigned 32 -> Bool -> Unsigned 64 -> Unsigned 4 -> MemRequest
mkMemRequest a w d c = MemRequest (Addr32 a) w (Data64 d) (ClientID4 c)

mkMemResponse :: Unsigned 64 -> Unsigned 4 -> Bool -> MemResponse
mkMemResponse d c v = MemResponse (Data64 d) (ClientID4 c) v

filterResp :: ClientID4 -> MemResponse -> Maybe MemResponse
filterResp cid resp
    | respClient resp == cid && respValid resp = Just resp
    | otherwise = Nothing

routeResponse :: ClientID4 -> Maybe MemResponse -> Maybe MemResponse
routeResponse cid maybeResp = maybeResp >>= filterResp cid

clientIdOf :: Index NumClients -> ClientID4
clientIdOf idx = ClientID4 (fromIntegral (fromEnum idx))

memoryArbiter
    :: HiddenClockResetEnable dom
    => Vec NumClients (Signal dom (Maybe MemRequest))
    -> Signal dom (Maybe MemResponse)
    -> (Signal dom (Maybe MemRequest), Vec NumClients (Signal dom (Maybe MemResponse)))
memoryArbiter clientReqs memResp = (memReqOut, clientResps)
  where
    (memReqOut, _grantVec) = unbundle (mealy arbiterT initialArbiterState (bundle clientReqs))
    clientResps = imap (\i _ -> fmap (routeResponse (clientIdOf i)) memResp) clientReqs

arbiterT
    :: (ArbiterState, Unsigned 8)
    -> Vec NumClients (Maybe MemRequest)
    -> ((ArbiterState, Unsigned 8), (Maybe MemRequest, Vec NumClients Bool))
arbiterT (ArbIdle, counter) reqs = case findIndex isJust reqs of
    Just idx ->
        let clientId = clientIdOf idx
            grant = imap (\i _ -> i == idx) reqs
        in ((ArbServing clientId 0, counter + 1), (reqs !! idx, grant))
    Nothing -> ((ArbIdle, counter), (Nothing, repeat False))
arbiterT (ArbServing client cycles, counter) _reqs
    | cycles < serviceCyclesVal - 1 =
        ((ArbServing client (cycles + 1), counter), (Nothing, repeat False))
    | otherwise = ((ArbIdle, counter), (Nothing, repeat False))

grantForRequestMask :: Unsigned 4 -> Signed 32
grantForRequestMask mask =
    let requests = imap
            (\i _ ->
                if testBit mask (fromEnum i)
                then Just (mkMemRequest 0 False 0 (fromIntegral (fromEnum i)))
                else Nothing)
            (repeat () :: Vec NumClients ())
    in case findIndex isJust requests of
        Just idx -> fromIntegral (fromEnum idx)
        Nothing -> -1

topEntity
    :: Clock System
    -> Reset System
    -> Enable System
    -> Vec NumClients (Signal System (Maybe MemRequest))
    -> Signal System (Maybe MemResponse)
    -> (Signal System (Maybe MemRequest), Vec NumClients (Signal System (Maybe MemResponse)))
topEntity = exposeClockResetEnable memoryArbiter

testInput :: Vec NumClients (Signal System (Maybe MemRequest))
testInput =
    pure (Just (mkMemRequest 0x1000 False 0 0))
        :> pure (Just (mkMemRequest 0x2000 True 0xDEADBEEF 1))
        :> pure Nothing
        :> pure Nothing
        :> Nil
