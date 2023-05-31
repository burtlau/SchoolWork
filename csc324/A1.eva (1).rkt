#lang racket ; CSC324 — 2023W — Assignment 1 — Eva Implementation

; Task: implement eva according to A1.eva-test.rkt.

(provide (contract-out (eva (any/c . -> . any/c)))
         ; Add any helper functions you tested in A1.eva-test.rkt.
         ; Whether you add contracts to them is optional.
         #;a-helper)

; Produce the value of a closed term from LCA.
(define (eva term) term)
