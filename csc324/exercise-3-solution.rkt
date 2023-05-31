#lang racket


; • CSC324 — 2023W — Exercise 3

; Due: Mon Feb 20th 5PM.

; Higher-order functions


; Implement X to produce the cartesian product of two lists.
; Write your answer here. 
; Half of the marks are for using higher-order functions
; to minimize the use of recursion. You may write helper functions.

(provide (contract-out 
                       (X (list? list? . -> . list?))))

(require rackunit)
(module+ test (require rackunit))

; · cartesian-products

(module+ test
  (check-equal?  (X '(a b c) '(d e f)) '((a d) (a e) (a f)
(b d) (b e) (b f)
(c d) (c e) (c f))))

(define (X list1 list2)
  (append-map
             (compose1 (curryr map (map list list2)) (curry cons)) list1))