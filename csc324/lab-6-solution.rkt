#lang racket

; • CSC324 — 2023W — Lab 6 — Solutions

; Recursion in the Lambda Calculus : Deriving the Z Combinator.

(require rackunit)

; Here is a recursive factorial definition.
(define F (λ (n) (if (zero? n) 1 (* n (F (sub1 n))))))
(check-equal? (F 5) 120)

; Recall: a let is shorthand for immediately making and calling a function ...
#;(let ((variable value) ...)
    body)
#;((λ (variable ...) body)
   value ...)
; ... and so can't reference the variables in their values which is why ...
#;(let ((F.1 (λ (n) (if (zero? n) 1 (* n (F.1 (sub1 n)))))))
    (F.1 5))
; ... fails to compile.

; Task: write out the corresponding make-and-call expression and confirm
; that it does not compile, then comment it out.

#;((λ (F.1) (F.1 5))
   (λ (n) (if (zero? n) 1 (* n (F.1 (sub1 n))))))

; The trick is to add a parameter to tell the function about itself, which
; it passes along to itself at each recursive call as well.
(define (F.1 me n) (if (zero? n) 1 (* n (me me (sub1 n)))))
(check-equal? (F.1 F.1 5) 120)

; This solves the problem, but requires altering the body of F which can be
; undesirable theoretically (i.e. to study the properties of LC, recursion,
; etc) and/or practically undesirable. We'll eliminate that and derive the
; Z Combinator (the eager evaluation version of the famous Y Combinator).

; Task: define F.2 in curried form: as a unary function (taking me), that
; produces a unary function to then take n.
(module+ test (check-equal? ((F.2 F.2) 5) 120))
(define F.2 (λ (me) (λ (n) (if (zero? n) 1 (* n ((me me) (sub1 n)))))))

; Task: replace F.2 in the following with its definition.
(check-equal?
 ((let ((F.3 #;F.2
             (λ (me) (λ (n) (if (zero? n) 1 (* n ((me me) (sub1 n))))))))
    (F.3 F.3))
  5)
 120)

; Task: to undo the the explicit passing of me in the body, make a local
; variable meme with value (me me) and use it. Then run the result and
; confirm that it's an infinite recursion: why?
#;(check-equal?
   ((let ((F.3 (λ (me)
                 (let (#;???
                       (meme (me me)))
                   (λ (n) #;???
                     (if (zero? n) 1 (* n (meme (sub1 n)))) )))))
      (F.3 F.3))
    5)
   120)

; Task: delay evaluation of (me me) to when it's actually used, by
; wrapping it in a lambda that behaves identically.
(check-equal?
 ((let ((F.3 (λ (me)
               (let ((meme (λ (n) #;???
                             ((me me) n))))
                 (λ (n) (if (zero? n) 1 (* n (meme (sub1 n)))) )))))
    (F.3 F.3))
  5)
 120)

; Task: call F.4 (which has the same body as the original F), to produce
; the innermost λ.
(define F.4 (λ (F) (λ (n) (if (zero? n) 1 (* n (F (sub1 n)))))))
(check-equal?
 ((let ((F.3 (λ (me) (let ((meme (λ (n) ((me me) n))))
                       (F.4 #;??? meme)))))
    (F.3 F.3))
  5)
 120)

; Task: the entire let is now a function of F.4: define that function!
(module+ test (check-equal?
               ((Z F.4)
                5)
               120))

(define (Z F)
  #;(let ???
      ???)
  (let ((F.3 (λ (me) (let ((meme (λ (n) ((me me) n))))
                       (F meme)))))
    (F.3 F.3)))

; Congratulations, you've defined the Z combinator!
(define ! (Z (λ (F) (λ (n) (if (zero? n) 1 (* n (F (sub1 n))))))))
(check-equal? (! 10) 3628800)
