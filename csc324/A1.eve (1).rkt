#lang racket ; CSC324 — 2023W — Assignment 1 — Eve Implementation

; Task: implement eve according to A1.eve-test.rkt.

(provide
 (contract-out (eve (any/c . -> . (list/c any/c (hash/c symbol? list?)))))
 ; Add any helper functions you tested in A1.eva-test.rkt.
 ; Whether you add contracts to them is optional.
 #;a-helper)

; · Support: indexer

; A constructor for a zero-arity function that when called successively
; produces symbols of the form prefix0 prefix1 prefix2 etc.

(provide (contract-out (indexer (any/c . -> . (-> symbol?)))))

(define (indexer prefix)
  (define last-index -1)
  (λ ()
    (local-require (only-in racket/syntax format-symbol))
    (set! last-index (add1 last-index))
    (format-symbol "~a~a" prefix last-index)))

; · eve

; Produce a two-element list with the value and the environment-closure table
; from evaluating an LCE term.
(define (eve term)

  ; A mutable table of environments and closures.
  ; Task: Look up hash-ref and hash-set! in the racket documentation.
  (define environments-closures (make-hash))
  
  ; Iterators to produce indices for environments and closures.
  (define En (indexer 'E))
  (define λn (indexer 'λ))

  ; Task: complete rec.
  (define (rec t E) t)
  
  (list (rec term (En))
        (make-immutable-hash (hash->list environments-closures))))
