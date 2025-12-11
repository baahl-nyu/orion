package main

//#include <stdlib.h>
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
)

func convertCIntToInt(v C.int) int {
	return int(v)
}
func convertCFloatToFloat(v C.float) float64 {
	return float64(v)
}

func CArrayToByteSlice(dataPtr unsafe.Pointer, length uint64) []byte {
	return unsafe.Slice((*byte)(dataPtr), length)
}

func CCharArrayToByteSlice(dataPtr *C.char, length C.ulong) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(dataPtr)), int(length))
}

func convertFloatToCFloat(v float64) C.float {
	return C.float(v)
}

func convertFloat64ToCDouble(v float64) C.double {
	return C.double(v)
}

func convertIntToCInt(v int) C.int {
	return C.int(v)
}

func convertULongtoCULong(v uint64) C.ulong {
	return C.ulong(v)
}

func convertUint64ToCULonglong(v uint64) C.ulonglong {
	return C.ulonglong(v)
}

func convertULongtoInt(v uint64) C.int {
	return C.int(v)
}

func convertByteToCChar(b byte) C.char {
	return C.char(b)
}

func CArrayToSlice[T, U any](ptr *U, length C.int, conv func(U) T) []T {
	cSlice := unsafe.Slice(ptr, int(length))
	result := make([]T, int(length))
	for i, v := range cSlice {
		result[i] = conv(v)
	}
	return result
}

func SliceToCArray[T, U any](slice []T, conv func(T) U) (*U, C.ulong) {
	n := len(slice)
	if n == 0 {
		return nil, 0
	}
	size := C.size_t(n) * C.size_t(unsafe.Sizeof(*new(U)))
	ptr := C.malloc(size)
	if ptr == nil {
		panic("C.malloc failed")
	}
	cArray := unsafe.Slice((*U)(ptr), n)
	for i, v := range slice {
		cArray[i] = conv(v)
	}
	return (*U)(ptr), C.ulong(n)
}

// Keys returns a slice of keys from the provided map.
func GetKeysFromMap[K comparable, V any](m map[K]V) []K {
	keys := make([]K, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

// Values returns a slice of values from the provided map.
func GetValuesFromMap[K comparable, V any](m map[K]V) []V {
	values := make([]V, 0, len(m))
	for _, v := range m {
		values = append(values, v)
	}
	return values
}

//export FreeCArray
func FreeCArray(ptr unsafe.Pointer) {
	C.free(ptr)
}

func PrintCipher(scheme Scheme, ctxt *rlwe.Ciphertext) {
	msg := make([]float64, ctxt.Slots())

	// Decode and check result
	ptxt := scheme.Decryptor.DecryptNew(ctxt)
	_ = scheme.Encoder.Decode(ptxt, msg)

	for i := 0; i < min(16, ctxt.Slots()); i++ {
		fmt.Printf("msg[%d]: %.5f\n", i, msg[i])
	}
}
