package main

import (
	"C"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
)

var (
	ptHeap = NewHeapAllocator()
	ctHeap = NewHeapAllocator()
)

func PushPlaintext(plaintext *rlwe.Plaintext) int {
	return ptHeap.Add(plaintext)
}

func PushCiphertext(ciphertext *rlwe.Ciphertext) int {
	return ctHeap.Add(ciphertext)
}

func RetrievePlaintext(plaintextID int) *rlwe.Plaintext {
	return ptHeap.Retrieve(plaintextID).(*rlwe.Plaintext)
}

func RetrieveCiphertext(ciphertextID int) *rlwe.Ciphertext {
	return ctHeap.Retrieve(ciphertextID).(*rlwe.Ciphertext)
}

// ---------------------------------------- //
//             PYTHON BINDINGS              //
// ---------------------------------------- //

//export DeletePlaintext
func DeletePlaintext(plaintextID C.int) {
	ptHeap.Delete(int(plaintextID))
}

//export DeleteCiphertext
func DeleteCiphertext(ciphertextID C.int) {
	ctHeap.Delete(int(ciphertextID))
}

//export GetPlaintextScale
func GetPlaintextScale(plaintextID C.int) C.ulong {
	plaintext := RetrievePlaintext(int(plaintextID))
	scaleBig := &plaintext.Scale.Value
	scale, _ := scaleBig.Uint64()
	return C.ulong(scale)
}

//export GetCiphertextScale
func GetCiphertextScale(ciphertextID C.int) C.ulong {
	ciphertext := RetrieveCiphertext(int(ciphertextID))
	scaleBig := &ciphertext.Scale.Value
	scale, _ := scaleBig.Uint64()
	return C.ulong(scale)
}

//export SetPlaintextScale
func SetPlaintextScale(plaintextID C.int, scale C.ulong) {
	plaintext := RetrievePlaintext(int(plaintextID))
	plaintext.Scale = rlwe.NewScale(uint64(scale))
}

//export SetCiphertextScale
func SetCiphertextScale(ciphertextID C.int, scale C.ulong) {
	ciphertext := RetrieveCiphertext(int(ciphertextID))
	ciphertext.Scale = rlwe.NewScale(uint64(scale))
}

//export GetPlaintextLevel
func GetPlaintextLevel(plaintextID C.int) C.int {
	plaintext := RetrievePlaintext(int(plaintextID))
	return C.int(plaintext.Level())
}

//export GetCiphertextLevel
func GetCiphertextLevel(ciphertextID int) C.int {
	ciphertext := RetrieveCiphertext(ciphertextID)
	return C.int(ciphertext.Level())
}

//export GetPlaintextSlots
func GetPlaintextSlots(plaintextID int) C.int {
	plaintext := RetrievePlaintext(plaintextID)
	slots := 1 << plaintext.LogDimensions.Cols
	return C.int(slots)
}

//export GetCiphertextSlots
func GetCiphertextSlots(ciphertextID int) C.int {
	ciphertext := RetrieveCiphertext(ciphertextID)
	slots := 1 << ciphertext.LogDimensions.Cols
	return C.int(slots)
}

//export GetCiphertextDegree
func GetCiphertextDegree(ciphertextID int) C.int {
	ciphertext := RetrieveCiphertext(ciphertextID)
	return C.int(ciphertext.Degree())
}

//export GetModuliChain
func GetModuliChain() (*C.ulong, C.ulong) {
	moduli := scheme.Params.Q()
	arrPtr, length := SliceToCArray(moduli, convertULongtoCULong)
	return arrPtr, length
}

//export GetAuxModuliChain
func GetAuxModuliChain() (*C.ulong, C.ulong) {
	moduli := scheme.Params.P()
	arrPtr, length := SliceToCArray(moduli, convertULongtoCULong)
	return arrPtr, length
}

//export GetLivePlaintexts
func GetLivePlaintexts() (*C.int, C.ulong) {
	ids := ptHeap.GetLiveKeys()
	arrPtr, length := SliceToCArray(ids, convertIntToCInt)
	return arrPtr, length
}

//export GetLiveCiphertexts
func GetLiveCiphertexts() (*C.int, C.ulong) {
	ids := ctHeap.GetLiveKeys()
	arrPtr, length := SliceToCArray(ids, convertIntToCInt)
	return arrPtr, length
}
