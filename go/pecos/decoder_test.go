// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package pecos

import (
	"fmt"
	"testing"
)

// TrivialDecoder is a toy decoder that XORs all syndrome bytes into one observable.
// This is what a Go decoder author's code looks like -- just implement the interface.
type TrivialDecoder struct {
	checks int
	bits   int
}

func (d *TrivialDecoder) Decode(syndrome []byte) (*DecodingResult, error) {
	if len(syndrome) != d.checks {
		return nil, fmt.Errorf("expected %d syndrome bytes, got %d", d.checks, len(syndrome))
	}

	observable := make([]byte, d.bits)
	// Trivial "decoding": XOR all syndrome bits into observable[0]
	var xor byte
	for _, b := range syndrome {
		xor ^= b
	}
	if d.bits > 0 {
		observable[0] = xor
	}

	converged := true
	return &DecodingResult{
		Observable: observable,
		Weight:     1.0,
		Converged:  &converged,
	}, nil
}

func (d *TrivialDecoder) CheckCount() int { return d.checks }
func (d *TrivialDecoder) BitCount() int   { return d.bits }

func TestDecoderInterface(t *testing.T) {
	d := &TrivialDecoder{checks: 4, bits: 2}

	// Verify it satisfies the Decoder interface
	var _ Decoder = d

	result, err := d.Decode([]byte{0, 1, 0, 1})
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if result.Observable[0] != 0 {
		t.Errorf("expected observable[0]=0 (1^1=0), got %d", result.Observable[0])
	}
	if result.Weight != 1.0 {
		t.Errorf("expected weight=1.0, got %f", result.Weight)
	}
}

func TestDecoderRegistration(t *testing.T) {
	d := &TrivialDecoder{checks: 3, bits: 1}

	handle := RegisterDecoder(d)
	if handle == nil {
		t.Fatal("RegisterDecoder returned nil")
	}
	defer handle.Destroy()

	if handle.Handle() == 0 {
		t.Error("handle should be non-zero")
	}

	// Verify the registered decoder still works via the registry
	looked := lookupDecoder(handle.Handle())
	if looked == nil {
		t.Fatal("lookupDecoder returned nil")
	}
	if looked.CheckCount() != 3 {
		t.Errorf("expected CheckCount=3, got %d", looked.CheckCount())
	}
}

func TestDecoderErrorHandling(t *testing.T) {
	d := &TrivialDecoder{checks: 4, bits: 2}

	// Wrong syndrome length should return error
	_, err := d.Decode([]byte{0, 1})
	if err == nil {
		t.Error("expected error for wrong syndrome length")
	}
}
