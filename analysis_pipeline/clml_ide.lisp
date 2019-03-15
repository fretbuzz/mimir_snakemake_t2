#!/usr/local/bin/sbcl --script

;; okay, let's input the parameters now!!!!
;; what needs to be turned into parameters??
;; (1) the csv file with the column-ized edgefile []
;; (2) the number of columns in the aforementioned file []
;; (3) the "averaging window"
;; (4) the length (timesteps)
;; (5) output file location.

(format t "~&~S~&" *posix-argv*)
(format t "~&~S~&" (car (cdr *posix-argv*)))

(defvar csv-file)
(setq csv-file (car (cdr *posix-argv*))) ;; e.g. "/Volumes/exM2/experimental_data/wordpress_info/wordpress_thirteen_t2/graphs/test_ide.csv"
(print csv-file)
(defvar num-cols)
(setq num-cols (parse-integer (car (cdr (cdr *posix-argv*)))))  ;; e.g. 1191
(defvar window-size)
(setq window-size (parse-integer (car (cdr (cdr (cdr *posix-argv*)))))) ;; e.g. 12
(defvar total-length-timesteps)
(setq total-length-timesteps (parse-integer (car (cdr (cdr (cdr (cdr *posix-argv*))))))) ;; 720
(defvar output-file-loc)
(setq output-file-loc (car (cdr (cdr (cdr (cdr (cdr *posix-argv*)))))))  ;;"./ide_clml_test.txt"



(load "~/quicklisp/setup.lisp")
(setf *read-default-float-format* 'double-float)
(ql:quickload :clml :verbose t)
(in-package :CLML.TIME-SERIES.ANOMALY-DETECTION)


(defvar relevantData)
(setq  relevantData (time-series-data (read-data-from-file csv-file
    :type :csv :csv-type-spec (make-list num-cols :initial-element 'double-float))))
(print relevantData)

(defvar results)

(setq results (loop with detector = (make-db-detector (sub-ts relevantData :start '(1 1) :end '(window-size num-cols)))
	for p across (ts-points (sub-ts relevantData :start '((+ window-size 1) 1) :end '(total-length-timesteps num-cols)))
	collect (funcall detector (ts-p-pos p))))

(with-open-file (str output-file-loc :direction :output :if-exists :supersede  :if-does-not-exist :create)
	(format str "~A~%" stuff))