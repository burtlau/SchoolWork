#lang racket 

; • CSC324 — 2023W — Exercise 2 — More tests for tails


; tails function.
(require "exercise-2.rkt")

(require rackunit)
(module+ test (require rackunit))

(check-equal? (tails "hi") (list "hi"))
(check-equal? (tails '(5)) (list '(5)))
(check-equal? (tails '(x)) (list '(x)))
(check-equal? (tails '(+ 1 2)) (list '(+ 1 2)))
(check-equal? (tails '(f 1 (if cond x y))) (list '(f 1 (if cond x y))))
(check-equal? (tails '(and #false (- 5))) (list '(- 5)))
(check-equal? (tails '(and (f 1) (g 2))) (list '(g 2)))
(check-equal? (tails '(and (f 1 2) (and "hi" (g 3 4)))) (list '(g 3 4)))
