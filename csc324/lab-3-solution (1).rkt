#lang racket

; • CSC324 — 2023W — Lab 3
; Quoting
; Point-free Style
; Simple Environment Substitution

; · This Lab's Steps

; YOU MIGHT NOT COMPLETE ALL THE STEPS DURING THE LAB.
; We will publish solutions afterwards.

(require rackunit)
(module+ test (require rackunit))

; · Quoting

; Recall: literals are pieces of *source code* representing values
; known/fixed/constant/specified at *compile time*.
; Single-quoting an identifier in source code makes it a symbol literal.

; Uncomment, and replace each “replace-me” with the expected literal result.
(check-equal? (let ((x 324)) "x") #;replace-me "x")
(check-equal? (let ((x 324)) 'x) #;replace-me 'x)
(check-equal? ((λ (x) "x") 324) #;replace-me "x")
(check-equal? ((λ (x) 'x) 324) #;replace-me 'x)

; (Single-)quoting a boolean, numeric, or string literal is redundant.
; Uncomment, and simplify the second expression where possible.
(check-equal? '#t #;'#t #t)
(check-equal? '324 #;'324 324)
(check-equal? "'hi" "'hi")
(check-equal? '"hi" #;'"hi" "hi")

; Quoting a quote however is not redundant (we'll discuss the meaning later).
(check-not-equal? ''hi 'hi)

; Quoting a parenthesized expression denotes a list, with the elements as if
; the components were quoted. We can also compute that list at run-time by
; calling the list function.
(check-equal? '(#t 324 "hi" hey)
              (list '#t '324 '"hi" 'hey))
; Remove any redundant quotes from the test's second argument.
(check-equal? '(#t 324 "hi" hey)
              #;(list '#t '324 '"hi" 'hey)
              (list #t 324 "hi" 'hey))

; For quoted literals with nested parentheses the meaning is recursive.
; Show this step-by-step by replacing the second argument in the three tests.
; A single step is:
; 1. Remove any redundant quotation.
; 2. Change
#;'(e1 e2 ...) ; to
#;(list 'e1 'e2 ...)

(check-equal? '((#t 324) () ((hi) hey))
              #;'((#t 324) () ((hi) hey))
              (list '(#t 324) '() '((hi) hey)))
(check-equal? '((#t 324) () ((hi) hey))
              #;'((#t 324) () ((hi) hey))
              (list (list '#t '324) (list) (list '(hi) 'hey)))
(check-equal? '((#t 324) () ((hi) hey))
              #;'((#t 324) () ((hi) hey))
              (list (list #t 324) (list) (list (list 'hi) 'hey)))

; · Point-free Style

; Define fix-1st as suggested by the test cases.
; Hint: what is its arity, what is the datatype of its result?
(module+ test
  (check-equal? ((fix-1st + 324) 1000) (+ 324 1000))
  (check-equal? ((fix-1st list 324) 1000) (list 324 1000))
  #;(check-equal? ((fix-1st bf v1) v2) (bf v1 v2)))

(define (fix-1st bf v1)
  #;(void)
  (λ (v2) (bf v1 v2)))

; Define Equal? using fix-1st.
; It takes a value v and produces a unary function that determines whether its
; argument is equal to v.

#;
(module+ test
  (check-true ((Equal? 324) 324))
  (check-false ((Equal? 324) 207)))

(define (Equal? v)
  #;(void)
  (fix-1st equal? v))

; · A Lookup-Table

; A table will be a list of two-element lists:
#;(list (list key value) (list key value) ...)

; Define unary function key-is? that takes a key k and produces a unary
; predicate that determines whether a two-element list
#;(list key value)
; has key equal to k. Use compose and Equal?.

(define (key-is? key)
  #;(void)
  (compose (Equal? key) first))

; Define values that takes a table and a key, and produces the list of all
; values for that key. Use key-is?, filter, and second.

(define (values table key)
  #;(void)
  (map second (filter (key-is? key) table)))

(module+ test
  (check-equal? (values '((1 "one") (2 "two") (1 "three")) 1)
                '("one" "three")))

#;(define (has-key? table key) (not (empty? (values table key))))

; Racket's compose takes any number of unary functions followed by a
; function f with arbitrary arity, and produces the composition of
; the functions (which then has the same arity as f).
#;(compose uf1 uf2 ... ufn f)
#;(λ (x ...) (uf1 (uf2 (... (ufn (f x ...))))))

; Use compose and values to define binary functions has-key? and first-value
; respectively, that take a table and a key and determine whether the table
; has that key and (assume it does) the first value for that key, respectively.
(define has-key?
  #;void
  (compose not empty? values))
(define first-value
  #;void
  (compose first values))

; · Substitution

; Complete sub, which takes a table and a value, and replaces every occurence
; of a key in the value with its value from the table.
; Then re-implement sub without using a helper function, replacing the recursive
; use of rec with
#;(fix-1st sub env)

(module+ test
  (check-equal? (sub '((one 1) (two 2) (one 3))
                     '(one two three (two three one)))
                '(1 2 three (2 three 1))))

(define (sub env e)
  #;(define (rec e)
      (cond ((list? e) (map rec e))
            ((has-key? env e) (first-value env e))
            (else e)))
  #;(rec e)
  (cond ((list? e) (map (fix-1st sub env) e))
        ((has-key? env e) (first-value env e))
        (else e)))
