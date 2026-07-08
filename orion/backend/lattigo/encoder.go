package main

import (
	"C"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

//export NewEncoder
func NewEncoder() {
	scheme.Encoder = ckks.NewEncoder(*scheme.Params)
}

//export Encode
func Encode(
	valuesPtr *C.float,
	lenValues C.int,
	level C.int,
	scale C.ulong,
) C.int {
	values := CArrayToSlice(valuesPtr, lenValues, convertCFloatToFloat)
	plaintext := ckks.NewPlaintext(*scheme.Params, int(level))
	plaintext.Scale = rlwe.NewScale(uint64(scale))

	scheme.Encoder.Encode(values, plaintext)

	idx := PushPlaintext(plaintext)
	return C.int(idx)
}

//export Decode
func Decode(
	plaintextID C.int,
) (*C.float, C.ulong) {
	plaintext := RetrievePlaintext(int(plaintextID))
	result := make([]float64, scheme.Params.MaxSlots())
	scheme.Encoder.Decode(plaintext, result)

	arrPtr, length := SliceToCArray(result, convertFloatToCFloat)
	return arrPtr, length
}

// EncodeComplex encodes real/imaginary lane pairs into CKKS slots. The
// input is an interleaved [re0, im0, re1, im1, ...] float array so that
// both lanes of each complex slot can be filled (as opposed to Encode,
// which only populates the real lane and leaves the imaginary lane zero).
//
//export EncodeComplex
func EncodeComplex(
	valuesPtr *C.float,
	lenValues C.int,
	level C.int,
	scale C.ulong,
) C.int {
	raw := CArrayToSlice(valuesPtr, lenValues, convertCFloatToFloat)

	values := make([]complex128, len(raw)/2)
	for i := range values {
		values[i] = complex(raw[2*i], raw[2*i+1])
	}

	plaintext := ckks.NewPlaintext(*scheme.Params, int(level))
	plaintext.Scale = rlwe.NewScale(uint64(scale))

	scheme.Encoder.Encode(values, plaintext)

	idx := PushPlaintext(plaintext)
	return C.int(idx)
}

// DecodeComplex returns the interleaved [re0, im0, re1, im1, ...] slot
// values of a plaintext encoded with EncodeComplex.
//
//export DecodeComplex
func DecodeComplex(
	plaintextID C.int,
) (*C.float, C.ulong) {
	plaintext := RetrievePlaintext(int(plaintextID))
	result := make([]complex128, scheme.Params.MaxSlots())
	scheme.Encoder.Decode(plaintext, result)

	interleaved := make([]float64, 2*len(result))
	for i, v := range result {
		interleaved[2*i] = real(v)
		interleaved[2*i+1] = imag(v)
	}

	arrPtr, length := SliceToCArray(interleaved, convertFloatToCFloat)
	return arrPtr, length
}
