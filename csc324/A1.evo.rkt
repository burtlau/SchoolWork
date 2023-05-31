#lang racket ; CSC324 — 2023W — Assignment 1 — Evo Implementation

; Task: implement evo according to A1.evo-test.rkt.

(provide
 (contract-out (evo (any/c . -> . (list/c any/c (hash/c symbol? list?)))))
 ; Add any helper functions you tested in A1.evo-test.rkt.
 ; Whether you add contracts to them is optional.
 #;a-helper)

; · Support: indexer and Void

(provide (contract-out (indexer (any/c . -> . (-> symbol?)))))

(define (indexer prefix)
  (define last-index -1)
  (λ ()
    (local-require (only-in racket/syntax format-symbol))
    (set! last-index (add1 last-index))
    (format-symbol "~a~a" prefix last-index)))

; There is no literal nor variable for the void value (although it has a printed
; representation when printed inside compound values), so we'll name it here and
; and also export it for testing.

(provide (contract-out (Void void?)))

(define Void (void))

; · evo

; Produce a two-element list with the value and the environment-closure table
; from evaluating an LCO term.
(define (evo term)

  ; A mutable table of environments and closures.
  (define environments-closures (make-hash))
  
  ; Iterators to produce indices for environments and closures.
  (define En (indexer 'E))
  (define λn (indexer 'λ))

  ; In-order evaluation of each body term during an enclosing evaluation of a lambda function call in evo
  (define (rec-eval body_terms E)
    (cond [(empty? body_terms) E]
          [else (rec-eval (rest body_terms) (rec (first body_terms) E))]))

  ; Check what symbol represent in environment closure.
  (define (get-var var E)
    (match E
      [(list(list a b) c) #:when(equal? a var) b]
      [(list(list a b) c) (get-var var (hash-ref environments-closures c))]))

  ; Get value by key
  (define (get-env key)
    (hash-ref environments-closures key))

  ; Get value by key
  (define (set-env key var)
    (hash-set! environments-closures key var))
  
  ; Task: complete rec.
  (define (rec t E) (match t
                      ; Case symbol
                      [(? symbol? t) (unbox (get-var t (get-env E)))]
                      ; Case lambda function
                      [(list 'λ a ..)
                         (define update_closure (λn)) 
                         (set-env update_closure (list t E)) update_closure]
                      ; Case function with input
                      [(list a b) #:when (list? a)
                         (define ev-a (get-env (rec a E)))
                         (define update_env (En))
                         (match ev-a [(list (list 'λ (list c) d ...) e)
                                      (set-env update_env (list (list c (box(rec b E))) e))
                                      (if (= (length d) 1)
                                          (rec (first d) update_env)
                                          (rec (last d)(rec-eval (drop-right d 1) update_env)))]
                                     ) ]
                      ; Case function with procedure
                      [(list a b) #:when (procedure? a)
                         (a (rec b E))]

                      ; Case assignment
                      [(list 'set! a b)
                       (set-box! (get-var a (get-env E)) (rec b E)) Void]

                      ; Case literal
                      [_ t]
                        ))
  
  (list (rec term (En))
        (make-immutable-hash (hash->list environments-closures))))
