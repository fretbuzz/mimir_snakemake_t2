#!/usr/local/bin/sbcl --script

(load "./asdf.lisp")

;; okay, let's input the parameters now!!!!
;; what needs to be turned into parameters??
;; (1) the csv file with the column-ized edgefile []
;; (2) the number of columns in the aforementioned file []
;; (3) the "averaging window"
;; (4) the length (timesteps)
;; (5) output file location.

(format t "~&~S~&" *posix-argv*)
(format t "~&~S~&" (car (cdr *posix-argv*)))

;;(defvar csv-file)
;;(setq csv-file (car (cdr *posix-argv*))) ;; e.g. "/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t2/graphs/test_ide.csv"
;;(print csv-file)
;;(defvar num-cols)
;;(setq num-cols (parse-integer (car (cdr (cdr *posix-argv*)))))  ;; e.g. 1191
;;(defvar window-size)
;;(setq window-size (parse-integer (car (cdr (cdr (cdr *posix-argv*)))))) ;; e.g. 12
;;(defvar total-length-timesteps)
;;(setq total-length-timesteps (parse-integer (car (cdr (cdr (cdr (cdr *posix-argv*))))))) ;; 720
;;(defvar output-file-loc)
;;(setq output-file-loc (car (cdr (cdr (cdr (cdr (cdr *posix-argv*)))))))  ;;"./ide_clml_test.txt"

(load "~/quicklisp/setup.lisp")
(setf *read-default-float-format* 'double-float)
(ql:quickload :clml :verbose t)
(in-package :CLML.TIME-SERIES.ANOMALY-DETECTION)

(defvar in)
(setq in (open "./clml_ide_params.txt" :if-does-not-exist nil))
(print "inOpened")

(defvar csv-file)
(setq csv-file (read-line in)) ;; e.g. "/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t2/graphs/test_ide.csv"
(print "csvFileSet")
(print csv-file)
(defvar num-cols)
(setq num-cols (parse-integer (read-line in)))  ;; e.g. 1191
(print "num-cols here")
(print num-cols)
(defvar window-size)
(setq window-size (parse-integer (read-line in))) ;; e.g. 12
(print "window-size here")
(print window-size)
(defvar total-length-timesteps)
(setq total-length-timesteps (parse-integer (read-line in))) ;; 720
(print "total-length-timesteps here")
(defvar output-file-loc)
(print total-length-timesteps)
(setq output-file-loc (read-line in))  ;;"./ide_clml_test.txt"
(print "all-vars-set")


(defmethod make-db-detector-without-online-thresh ((ts time-series-dataset)
                             &key beta (typical :svd) (pc 0.005d0) (normalize t))
  (assert (< 0d0 pc 1d0) (pc))
  (let* ((dim (length (dataset-dimensions ts)))
        (vecs (map 'list
                 (lambda (p) (let ((vec (ts-p-pos p)))
                              (if normalize (normalize-vec vec (copy-seq vec)) vec)))
                 (ts-points ts)))
         (moments (get-initial-moments vecs :typical-method typical)))
    (unless beta (setf beta (dfloat (/ (length vecs)))))
    (lambda (new-dvec)
      (assert (eql (length new-dvec) dim))
      (let* ((vec (if normalize (normalize-vec new-dvec (copy-seq new-dvec)) new-dvec))
             (typ (calc-typical-pattern vecs :method typical))
             (score (dissimilarity typ vec)))
        (setf moments (next-moments score moments beta)
              vecs (append (cdr vecs) (list vec)))
        (values score)))))



(defvar relevantData)
(setq  relevantData (time-series-data (read-data-from-file csv-file
    :type :csv :csv-type-spec (make-list num-cols :initial-element 'double-float))))
(print relevantData)

(defvar results)

(print "about to do ide portion")
(print (list (+ window-size 1) 1))

(setq results (loop with detector = (make-db-detector-without-online-thresh  (sub-ts relevantData :start '(1 1) :end (list window-size num-cols)))
	for p across (ts-points (sub-ts relevantData :start (list (+ window-size 1) 1) :end (list total-length-timesteps num-cols)))
	collect (funcall detector (ts-p-pos p))))

;;(setq results (loop with detector = (make-db-detector (sub-ts relevantData :start '(1 1) :end (list window-size num-cols)))
;;    for p across (ts-points (sub-ts relevantData :start (list (+ window-size 1) 1) :end (list total-length-timesteps num-cols)))
;;    collect (funcall detector (ts-p-pos p))))

(print "about to write results to file")

(with-open-file (str output-file-loc :direction :output :if-exists :supersede  :if-does-not-exist :create)
	(format str "~A~%" results))

