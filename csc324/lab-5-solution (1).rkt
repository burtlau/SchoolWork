#lang racket

; • CSC324 — 2023W — Lab 5

(require rackunit)

(define-syntax M
  (syntax-rules ()
    ((M n) (if (zero? n) n (M (- n 1))))))

; Why should we NOT try ...
#;(M 0)
; ★ Because *at compile time* it gets rewritten to ...
#;(if (zero? 0) 0 (M (- 0 1)))
; ... which has something of the form (M _) in it, so that gets rewritten to ...
#;(if (zero? 0) 0 (if (zero? (- 0 1)) (- 0 1) (M (- (- 0 1) 1))))
; ... which has something of the form (M _) in it, so that gets rewritten,
; and so on which never ends *compilation*.
; Syntax-rules macros do not run code, they don't even know what if, zero?, -,
; etc mean. They simply rearrange code based on form.
(define-syntax MM
  (syntax-rules ()
    ((MM X) (A (B X) X (MM (C X D))))))
#;(MM 0)
#;(A (B 0) 0 (MM (C 0 D)))
#;(A (B 0) 0 (A (B (C 0 D) (C 0 D) (MM (C (C 0 D) D)))))
; etc.


; · Class Macro

; Suppose we want to make a class operation, taking a constructor header
; and method names paired with their implementations ...

#;(class (Person surname given birth-year)
    (greeted (λ (with) (string-append with " " given " " surname ".")))
    (centennial (+ birth-year 100)))

; ... and produce the corresponding class/constructor ...

#;(define (Person surname given birth-year)
    (λ (msg)
      (match msg
        ('greeted (λ (with) (string-append with " " given " " surname ".")))
        ('centennial (+ birth-year 100)))))

#;(module+ test
    (check-equal? (((Person "Furler" "Sia" 1975) 'greeted) "Hi") "Hi Sia Furler.")
    (check-equal? ((Person "Lovelace" "Ada" 1852) 'centennial) 1952))

; It turns out that, unlike the double-quotes in string literals, single-quote
; is an operation (at compile time), and a macro can construct (at compile time)
; a symbol literal from single-quote and an identifier. In fact, single-quote
; is an abbreviation of a parenthesized use of the operation quote:
(check-equal? 'x (quote x))

; So we want ...
#;(class (class-name init-variable ...)
    (method-name method-body)
    ...)
; ... to mean ...
#;(define (class-name init-variable ...)
    (λ (msg)
      (match msg
        ((quote method-name) method-body)
        ...)))

; The syntax-rules pattern rewrite language interprets "..." as Kleene-*,
; both in the pattern and in the result, so we can just write ...
(define-syntax simpleclass
  (syntax-rules ()
    ((class (class-name init-variable ...)
       (method-name method-body)
       ...)
     (define (class-name init-variable ...)
       (λ (msg)
         (match msg
           ((quote init-variable) init-variable)
           ...
           ((quote method-name) method-body)
           ...
           (_ "unimplemented!")))))))

; So now comment out the earlier definitions of Person, and rely on
; simpleclass to make those definitions from ...

#;(simpleclass (Person surname given birth-year)
    (greeted (λ (with) (string-append with " " given " " surname ".")))
    (centennial (+ birth-year 100)))

#;(module+ test
  (define p (Person "A" "B" 324))
  (check-equal? (p 'surname) "A")
  (check-equal? (p 'given) "B")
  (check-equal? (p 'birth-year) 324)
  (check-equal? (p 'centennial) 424)
  (check-equal? ((p 'greeted) "Hi") "Hi B A."))

; · We want now to extend Our Class Macro

; Task: complete the implementation of Bank class with setters
; for the fields sb (saving balance) and cb (checking balance):

(module+ test  
  (define b (Bank 3 4))  
  (((b 'set) 'sb) 5) ; ★ not using the return value ...
  (check-equal? (b 'sb) 5) ; ... yet there's a change to sb
  (((b 'set) 'cb) 12)
  (check-equal? (b 'bal) 17))
#;
(define (Bank sb cb)
  (λ (tx)
    (match tx
      ('bal (+ sb cb))
      ('sb sb)
      ('cb cb)
      ('set (λ (balance) (λ (v) (match balance 
                                  ('sb (set! sb v))
                                  ('cb (set! cb v))))))
      ('writeCheck (λ (v) (Bank sb (- cb v))))
      ('transactSaving (λ (v) (Bank (+ sb v) cb))))))

; A macro class that extends simpleclass,
; to automatically create the setters necessary to implement Bank class ...
(define-syntax class
  (syntax-rules ()
    ((class (name init ...)
       (method-name method-body)
       ...)
     (define (name init ...)
       (λ (msg)
         (match msg
           ((quote init) init)
           ...
           ; ★ mimic the setters
           ('set (λ (field)
                   (λ (v) (match field
                            ; We put Kleene-* after exactly the part to repeat.
                            ((quote init) (set! init v))
                            ...))))
           ((quote method-name) method-body)
           ...))))))

; Comment out your definition of Bank above, and uncomment the following,
; then use the macro stepper to confirm it produces the same code as the
; version before you added the setter.
(class (Bank sb cb)
  (bal (+ sb cb))
  (writeCheck (λ (v) (Bank sb (- cb v))))
  (transactSaving (λ (v) (Bank (+ sb v) cb))))

; · Lazy Lists

; Recall this constructor and these deconstructors/accessors for lazy lists.
(define-syntax conz (syntax-rules () ((conz e r) (list e (λ () r)))))

(define firzt first)
(define (rezt lz) ((second lz)))

; Recall Map for infinite lazy lists.
(define (Map f lz) (conz (f (firzt lz)) (Map f (rezt lz))))

; Recall the infinite list of natural numbers.
(define N (conz 0 (Map add1 N)))

; Define nth that takes a lazy list and an index, and produces the element
; at that index.

(module+ test
  (check-equal? (nth N 324) 324)
  (check-equal? (nth (Map sqr N) 18) 324))

(define (nth lz n)
  ; ★
  (if (zero? n) (firzt lz) (nth (rezt lz) (- n 1))))
