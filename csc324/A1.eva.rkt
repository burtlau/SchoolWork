#lang racket ; CSC324 — 2023W — Assignment 1 — Eva Implementation

; Task: implement eva according to A1.eva-test.rkt.

(provide (contract-out (eva (any/c . -> . any/c)))
         ; Add any helper functions you tested in A1.eva-test.rkt.
         ; Whether you add contracts to them is optional.
         #;a-helper)


      

; Produce the value of a closed term from LCA.
(define (eva term)
  (define (helper-input func arg)
    (match func
      ; case closure function
      [(list a b c) #:when(and (not (list? c)) (not (symbol? c))) c]
      
      ; case argument and function body are equal
      [(list a b c) #:when(or (equal? (first b) c)) (match arg
                                                 [(list d e) (helper-input func (helper-input d e))]
                                                 [_ arg])]
      ; case argument and function body are not equal
      [(list a b c) #:when(not (equal? (first b) c)) (match c
                                                       [(list 'λ (list d) e)  (map (λ (z) (if (equal? z (first b))
                                                                                              (eva arg)
                                                                                              z)) c)]
                                                       [(list d e) #:when(list? d) (eva (list (list a b (eva c)) arg)) ]
                                                       [(list _ ...) (map (λ (z) (if (equal? z (first b))
                                                                                    (eva arg)
                                                                                     z)) c)]
                                                       [_  c]
                                                       [else #t])]
      ; case argument is lambda function
      [(list a b c) (match c
                      [(list d e f) c])]
      ; case func also needs to evaluate
      [(list a b)  (list (eva func) arg)]
      [else (last func)]
    )
   )
  ; eva base case
  (match term
    [(list a b c) term]
    [(list a b) (eva (helper-input a b))]
    [else term]))
