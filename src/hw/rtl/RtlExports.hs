{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE NoImplicitPrelude #-}

module RtlExports where

import Clash.Prelude
import qualified Prelude as P
import Foreign.C.Types (CDouble (..), CInt (..), CUInt (..), CULLong (..))
import Foreign.Marshal.Array (peekArray)
import Foreign.Ptr (Ptr)
import System.IO (IO)

import qualified MemoryArbiter as MA
import qualified RankerCore as RC
import qualified SSISearch as SSI

foreign export ccall jaide_rtl_abi_version :: CUInt

jaide_rtl_abi_version :: CUInt
jaide_rtl_abi_version = 1

foreign export ccall jaide_rtl_mix_hash :: CULLong -> CULLong -> CULLong

jaide_rtl_mix_hash :: CULLong -> CULLong -> CULLong
jaide_rtl_mix_hash state value =
    P.fromIntegral (toInteger (SSI.mixHash (P.fromIntegral state) (P.fromIntegral value)))

foreign export ccall jaide_rtl_count_bits64 :: CULLong -> CUInt

jaide_rtl_count_bits64 :: CULLong -> CUInt
jaide_rtl_count_bits64 value =
    P.fromIntegral (toInteger (SSI.countBits64 (P.fromIntegral value)))

foreign export ccall jaide_rtl_isqrt32 :: CUInt -> CUInt

jaide_rtl_isqrt32 :: CUInt -> CUInt
jaide_rtl_isqrt32 value =
    P.fromIntegral (toInteger (SSI.isqrt32 (P.fromIntegral value)))

foreign export ccall jaide_rtl_signature_similarity :: CULLong -> CULLong -> CUInt

jaide_rtl_signature_similarity :: CULLong -> CULLong -> CUInt
jaide_rtl_signature_similarity a b =
    P.fromIntegral (toInteger (SSI.signatureSimilarity (P.fromIntegral a) (P.fromIntegral b)))

foreign export ccall jaide_rtl_compute_similarity :: CULLong -> CULLong -> CUInt

jaide_rtl_compute_similarity :: CULLong -> CULLong -> CUInt
jaide_rtl_compute_similarity a b =
    P.fromIntegral (toInteger (SSI.computeSimilarity (P.fromIntegral a) (P.fromIntegral b)))

foreign export ccall jaide_rtl_bucket_index :: CULLong -> CUInt

jaide_rtl_bucket_index :: CULLong -> CUInt
jaide_rtl_bucket_index position =
    P.fromIntegral (fromEnum (SSI.bucketIndex (P.fromIntegral position)))

foreign export ccall jaide_rtl_hash_tokens :: Ptr CUInt -> CUInt -> IO CULLong

jaide_rtl_hash_tokens :: Ptr CUInt -> CUInt -> IO CULLong
jaide_rtl_hash_tokens tokenPtr tokenCount = do
    let count = P.fromIntegral tokenCount :: P.Int
    values <- peekArray count tokenPtr
    let folded = P.foldl
            (\acc value -> SSI.mixHash acc (resize (P.fromIntegral value :: Unsigned 32)))
            (SSI.mixHash 0 (P.fromIntegral count))
            values
    P.return (P.fromIntegral (toInteger folded))

foreign export ccall jaide_rtl_fuse_scores
    :: CDouble -> CDouble -> CDouble -> CDouble -> CDouble -> CDouble

jaide_rtl_fuse_scores :: CDouble -> CDouble -> CDouble -> CDouble -> CDouble -> CDouble
jaide_rtl_fuse_scores base overlap jaccard proximity diversity =
    let toFix :: CDouble -> RC.ScoreFix
        toFix (CDouble d) = RC.ScoreFix (realToFrac d)
        RC.ScoreFix fused = RC.fuseScores
            (toFix base)
            (toFix overlap)
            (toFix jaccard)
            (toFix proximity)
            (toFix diversity)
    in CDouble (realToFrac fused)

foreign export ccall jaide_rtl_arbiter_first_grant :: CUInt -> CInt

jaide_rtl_arbiter_first_grant :: CUInt -> CInt
jaide_rtl_arbiter_first_grant mask =
    P.fromIntegral (toInteger (MA.grantForRequestMask (P.fromIntegral mask)))

foreign export ccall jaide_rtl_arbiter_service_cycles :: CUInt

jaide_rtl_arbiter_service_cycles :: CUInt
jaide_rtl_arbiter_service_cycles = P.fromIntegral (toInteger MA.serviceCyclesVal)

foreign export ccall jaide_rtl_max_search_depth :: CUInt

jaide_rtl_max_search_depth :: CUInt
jaide_rtl_max_search_depth = P.fromIntegral (toInteger SSI.maxSearchDepthVal)
