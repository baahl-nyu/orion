package main

import (
	"C"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/polynomial"
	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/ring"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)
import (
	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/lintrans"
)

type Scheme struct {
	Params        *ckks.Parameters
	KeyGen        *rlwe.KeyGenerator
	SecretKey     *rlwe.SecretKey
	PublicKey     *rlwe.PublicKey
	RelinKey      *rlwe.RelinearizationKey
	EvalKeys      *rlwe.MemEvaluationKeySet
	Encoder       *ckks.Encoder
	Encryptor     *rlwe.Encryptor
	Decryptor     *rlwe.Decryptor
	Evaluator     *ckks.Evaluator
	PolyEvaluator *polynomial.Evaluator
	LinEvaluator  *lintrans.Evaluator
	Bootstrapper  *bootstrapping.Evaluator
}

var scheme Scheme

//export NewScheme
func NewScheme(
	logN C.int,
	logQPtr *C.int, lenQ C.int,
	logPPtr *C.int, lenP C.int,
	logScale C.int,
	h C.int,
	ringType *C.char,
	keysPath *C.char,
	ioMode *C.char,
) {
	// Convert LogQ and LogP to Go slices
	logQ := CArrayToSlice(logQPtr, lenQ, convertCIntToInt)
	logP := CArrayToSlice(logPPtr, lenP, convertCIntToInt)

	ringT := ring.Standard
	if C.GoString(ringType) != "standard" {
		ringT = ring.ConjugateInvariant
	}

	var err error
	var params ckks.Parameters

	if params, err = ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{
			LogN:            int(logN),
			LogQ:            logQ,
			LogP:            logP,
			LogDefaultScale: int(logScale),
			Xs:              ring.Ternary{H: int(h)},
			RingType:        ringT,
		}); err != nil {
		panic(err)
	}

	// Rotation keys and bootstrappers from a previous scheme were made
	// for its keys. Drop them so this scheme generates its own.
	DeleteRotationKeys()
	DeleteBootstrappers()

	keyGen := rlwe.NewKeyGenerator(params)

	scheme = Scheme{
		Params:        &params,
		KeyGen:        keyGen,
		SecretKey:     nil,
		PublicKey:     nil,
		RelinKey:      nil,
		EvalKeys:      nil,
		Encoder:       nil,
		Encryptor:     nil,
		Decryptor:     nil,
		Evaluator:     nil,
		PolyEvaluator: nil,
		LinEvaluator:  nil,
		Bootstrapper:  nil,
	}
}

//export DeleteScheme
func DeleteScheme() {
	scheme = Scheme{}

	DeleteRotationKeys()
	DeleteBootstrappers()
	DeleteMinimaxSignMap()

	ltHeap.Reset()
	polyHeap.Reset()
	ptHeap.Reset()
	ctHeap.Reset()
}
