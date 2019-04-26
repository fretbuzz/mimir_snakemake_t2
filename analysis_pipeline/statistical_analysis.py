import copy
import errno
import os
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import generate_heatmap, process_roc, generate_report
from jinja2 import FileSystemLoader, Environment

class single_model_stats_pipeline():
    def __init__(self, aggregate_mod_score_df, base_output_name, skip_model_part, clf, drop_pairwise_features, timegran,
                 lasso_feature_selection_p, dont_prepare_data_p=False, skip_heatmap_p=True,
                 no_labeled_data=False):

        self.no_labeled_data = no_labeled_data
        self.dont_prepare_data_p=dont_prepare_data_p
        self.method_name = 'ensemble'
        self.aggregate_mod_score_df = copy.deepcopy(aggregate_mod_score_df)
        self.base_output_name = base_output_name
        self.skip_model_part = skip_model_part
        self.clf = clf
        self.drop_pairwise_features = drop_pairwise_features
        self.time_gran = timegran
        self.time_gran_to_debugging_csv = {}
        self.lasso_feature_selection_p = lasso_feature_selection_p
        self.using_pretrained_model = False

        self.feature_activation_heatmaps = ['none.png']
        self.feature_raw_heatmaps = ['none.png']
        self.feature_activation_heatmaps_training = ['none.png']
        self.feature_raw_heatmaps_training =  ['none.png']
        self.method_to_test_predictions = {}
        self.method_to_train_predictions = {}

        #if not dont_prepare_data_p:
        self.X_train, self.y_train, self.X_test, self.y_test, self.pre_drop_X_train, self.time_gran_to_debugging_csv, \
        self.dropped_feature_list, self.ide_train, self.ide_test, self.exfil_weights_train, self.exfil_weights_test, \
        self.exfil_paths_test, self.exfil_paths_train, self.out_traffic, self.cilium_train, self.cilium_test = \
            prepare_data(self.aggregate_mod_score_df, self.skip_model_part, self.time_gran_to_debugging_csv,
                         self.time_gran, self.drop_pairwise_features)

        self.method_to_test_predictions['ide'] = self.ide_test
        self.method_to_train_predictions['ide'] = self.ide_train
        self.method_to_test_predictions['cilium'] = self.cilium_test
        self.method_to_train_predictions['cilium'] = self.cilium_train
        self.dropped_columns = list(self.pre_drop_X_train.columns.difference(self.X_train.columns))

        self.train_predictions = None
        self.test_predictions = None
        self.coef_dict  = None
        self.coef_feature_df = None
        self.model_params = None

        self.ROC_path = base_output_name + '_good_roc_'
        self.title = 'ROC Linear Combination of Features at ' + str(timegran)
        self.plot_name = 'sub_roc_lin_comb_features_' + str(timegran)
        self.debugging_csv_path = base_output_name + 'DEBUGGING_modz_feat_df_at_time_gran_of_'+ str(timegran) + '_sec.csv'
        self.plot_path = None

        self.method_to_test_thresholds = {}
        self.method_to_optimal_f1_scores_test, self.method_to_optimal_predictions_test = {},{}
        self.method_to_optimal_f1_scores_train, self.method_to_optimal_predictions_train = {},{}
        self.method_to_cm_df_train, self.method_to_cm_df_test = {},{}
        self.method_to_optimal_thresh_test, self.method_to_optimal_thresh_train = {}, {}
        self.skip_heatmaps=skip_heatmap_p

        try:
            os.makedirs('./temp_outputs')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


    def _prepare_data(self):
        pass # will want to move the prepare_data function over here... since it is cringy to have
        # it all in a seperate function with a bazillion parameters/returns (like above)

    def _generate_heatmap(self, training_p):
        if training_p:
            train_test = 'training_'
        else:
            train_test = 'test_'

        print "self.time_gran", self.time_gran
        if type(self.time_gran) == tuple:
            time_gran = "_".join([str(i) for i in self.time_gran]) + '_multi_'
        else:
            time_gran = str(self.time_gran)

        # make heatmaps so I can see which features are contributing
        current_heatmap_val_path = self.base_output_name + train_test + 'coef_val_heatmap_' + str(time_gran) + '.png'
        local_heatmap_val_path = 'temp_outputs/' + train_test + 'heatmap_coef_val_at_' +  str(time_gran) + '.png'
        current_heatmap_path = self.base_output_name + train_test + 'coef_act_heatmap_' + str(time_gran) + '.png'
        local_heatmap_path = 'temp_outputs/' + train_test + 'heatmap_coef_contribs_at_' +  str(time_gran) + '.png'

        if training_p:
            coef_impact_df, raw_feature_val_df = generate_heatmap.generate_covariate_heatmap(self.coef_dict, self.X_train, self.exfil_paths_train)
        else:
            coef_impact_df, raw_feature_val_df = generate_heatmap.generate_covariate_heatmap(self.coef_dict, self.X_test, self.exfil_paths_test)

        generate_heatmap.generate_heatmap(coef_impact_df, local_heatmap_path, current_heatmap_path)
        generate_heatmap.generate_heatmap(raw_feature_val_df, local_heatmap_val_path, current_heatmap_val_path)

        if training_p:
            self.feature_activation_heatmaps_training = ['../' + local_heatmap_path]
            self.feature_raw_heatmaps_training = ['../' + local_heatmap_val_path]
        else:
            self.feature_activation_heatmaps = ['../' + local_heatmap_path]
            self.feature_raw_heatmaps = ['../' + local_heatmap_val_path]
        print coef_impact_df

    def _generate_rocs(self):
        list_of_x_vals = []
        list_of_y_vals = []
        line_titles = []
        for method, test_predictions in self.method_to_test_predictions.iteritems():
            #print "self.y_test",self.y_test
            #print "test_predictions", test_predictions
            print "method", method
            #print "test_predictions", test_predictions
            test_predictions = np.nan_to_num(test_predictions)
            #print "self.y_test",self.y_test
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=self.y_test, y_score=test_predictions, pos_label=1)
            list_of_x_vals.append(fpr)
            list_of_y_vals.append(tpr)
            line_titles.append(method)
            self.method_to_test_thresholds[method] = thresholds

        print "list_of_x_vals", list_of_x_vals
        print "list_of_y_vals", list_of_y_vals
        print "roc_path", self.ROC_path + self.plot_name
        ax, _, plot_path = construct_ROC_curve(list_of_x_vals, list_of_y_vals, self.title, self.ROC_path + self.plot_name,
                                               line_titles, show_p=False)
        self.plot_path = plot_path

    def _generate_optimal_predictions(self, method_to_predictions, correct_y):
        method_to_optimal_f1_scores = {}
        method_to_optimal_predictions = {}
        method_to_optimal_thresh = {}
        for method, predictions in method_to_predictions.iteritems():
            thresholds = self.method_to_test_thresholds[method]
            ### determination of the optimal operating point goes here (take all the thresh vals and predictions,
            ### find the corresponding f1 scores (using sklearn func), and then return the best.
            optimal_f1_score, optimal_thresh = process_roc.determine_optimal_threshold(correct_y, predictions, thresholds)
            method_to_optimal_f1_scores[method] = optimal_f1_score
            optimal_predictions = [int(i > optimal_thresh) for i in predictions]
            method_to_optimal_predictions[method] = optimal_predictions
            method_to_optimal_thresh[method] = optimal_thresh
        return method_to_optimal_f1_scores, method_to_optimal_predictions, method_to_optimal_thresh

    def _generate_confusion_matrixes(self, method_to_optimal_predictions, correct_y, exfil_paths, exfil_weights):
        method_to_cm_df = {}
        for method, optimal_predictions in method_to_optimal_predictions.iteritems():
            print "correct_y", len(correct_y)
            print "optimal_predictions", len(optimal_predictions)
            confusion_matrix = determine_categorical_cm_df(correct_y, optimal_predictions, exfil_paths,
                                                                     exfil_weights)
            method_to_cm_df[method] = confusion_matrix
        return method_to_cm_df

    def _generate_debugging_csv(self):
        print "self.time_gran_to_debugging_csv[self.time_gran]", self.time_gran_to_debugging_csv[self.time_gran]
        print "self.train_predictions",self.train_predictions
        print "self.test_predictions", self.test_predictions
        print "np.concatenate([self.train_predictions, self.test_predictions])", np.concatenate(
            [self.train_predictions, self.test_predictions])
        print self.time_gran_to_debugging_csv[self.time_gran].shape
        print len(np.concatenate([self.train_predictions, self.test_predictions]))

        self.time_gran_to_debugging_csv[self.time_gran].loc[:, "aggreg_anom_score"] = np.concatenate(
            [self.train_predictions, self.test_predictions])

        # I don't want the attributes w/ zero coefficients to show up in the debugging csv b/c it makes it hard to read
        for feature,coef in self.coef_dict.iteritems():
            print "coef_check", coef, not coef, feature
            if not coef:
                print "just_dropped", feature
                try:
                    self.time_gran_to_debugging_csv[self.time_gran] = self.time_gran_to_debugging_csv[self.time_gran].drop([feature],axis=1)
                    self.coef_feature_df = self.coef_feature_df.drop(feature, axis=0)
                except:
                    pass
            for dropped_feature in self.dropped_feature_list + self.dropped_columns:
                try:
                    self.time_gran_to_debugging_csv[self.time_gran] = \
                        self.time_gran_to_debugging_csv[self.time_gran].drop([dropped_feature], axis=1)
                except:
                    pass

        self.time_gran_to_debugging_csv[self.time_gran].to_csv(self.debugging_csv_path , na_rep='?')

    # this function will find the cm that results from having <= a specified fp_limit
    # the point here is to fit allow the formation of the resulting model by decreasing the value
    # of the exfil rates along the various paths in a coherent way
    def find_optimal_cm_given_fps(self, fp_limit):
        fpr_limit = float(fp_limit) / len(self.y_test)
        fprs, tprs, thresholds = sklearn.metrics.roc_curve(y_true=self.y_test, y_score=self.test_predictions, pos_label=1)
        fprs_belowLimit = [fpr for fpr in fprs if fpr <= fpr_limit]
        max_fpr_index = np.argmax(fprs_belowLimit)
        max_fpr = fprs[max_fpr_index]
        max_threshold = thresholds[max_fpr_index]
        cur_predictions = [i >= max_threshold for i in self.test_predictions]
        confusion_matrix = determine_categorical_cm_df(self.y_test, cur_predictions,
                                                       self.exfil_paths_test, self.exfil_weights_test)
        return confusion_matrix

    def generate_report_section(self):
        if not self.skip_heatmaps:
            if  not self.using_pretrained_model:
                self._generate_heatmap(training_p=True)
            self._generate_heatmap(training_p=False)

        number_attacks_in_test = len(self.y_test[self.y_test['labels'] == 1])
        number_non_attacks_in_test = len(self.y_test[self.y_test['labels'] == 0])
        percent_attacks = (float(number_attacks_in_test) / (number_non_attacks_in_test + number_attacks_in_test))
        number_attacks_in_train = len(self.y_train[self.y_train['labels'] == 1])
        number_non_attacks_in_train = len(self.y_train[self.y_train['labels'] == 0])


        env = Environment(
            loader=FileSystemLoader(searchpath="./report_templates")
        )
        table_section_template = env.get_template("table_section.html")

        coef_feature_df = self.coef_feature_df.to_html()

        if not self.using_pretrained_model:
            attacks_found_training=self.method_to_cm_df_train[self.method_name].to_html()
            percent_attacks_train = (float(number_attacks_in_train) / (number_non_attacks_in_train + number_attacks_in_train))
            feature_activation_heatmap_training=self.feature_activation_heatmaps_training[0]
            feature_raw_heatmap_training=self.feature_raw_heatmaps_training[0]
        else:
            attacks_found_training= 'none.png'
            percent_attacks_train = 0.0
            feature_activation_heatmap_training='none.png'
            feature_raw_heatmap_training='none.png'


        report_section = table_section_template.render(
            time_gran=str(self.time_gran) + " sec granularity",
            roc=self.plot_path,
            feature_table=coef_feature_df,
            model_params=self.model_params,
            optimal_fOne=self.method_to_optimal_f1_scores_test[self.method_name],
            percent_attacks=percent_attacks,
            attacks_found=self.method_to_cm_df_test[self.method_name].to_html(),
            attacks_found_training=attacks_found_training,
            percent_attacks_training=percent_attacks_train,
            feature_activation_heatmap=self.feature_activation_heatmaps[0],
            feature_activation_heatmap_training=feature_activation_heatmap_training,
            feature_raw_heatmap=self.feature_raw_heatmaps[0],
            feature_raw_heatmap_training=feature_raw_heatmap_training,
            ideal_threshold=self.method_to_optimal_thresh_test[self.method_name]
        )

        return report_section

    def generate_model(self, using_pretrained_model=False, multi_time_train=False):
        if not using_pretrained_model:
            if not multi_time_train:
                self.clf.fit(self.X_train, self.y_train)
                self.train_predictions = self.clf.predict(X=self.X_train)
            else: # AM I SURE I WANT THIS
                ## fitting on the training set... so not super valid unless eval sete is there too...
                self.clf.fit(self.X_test, self.y_test)
                self.train_predictions = self.clf.predict(X=self.X_train)

            self.using_pretrained_model = False
        else:
            self.train_predictions = np.array([])
            self.using_pretrained_model = True

        print "Qt", self.time_gran
        print self.X_test.shape
        self.test_predictions = self.clf.predict(X=self.X_test)

        ## would this cause a problem on the eval set????
        self.coef_dict  = get_coef_dict(self.clf, self.X_train.columns.values, self.base_output_name, self.X_train.dtypes, sanity_check_number_coefs=(not using_pretrained_model))
        self.coef_feature_df = pd.DataFrame.from_dict(self.coef_dict, orient='index')
        self.coef_feature_df.columns = ['Coefficient']

        self.model_params = self.clf.get_params()
        try:
            self.model_params['alpha_val'] = self.clf.alpha_
        except:
            pass
        self.method_to_test_predictions[self.method_name] = self.test_predictions
        self.method_to_train_predictions[self.method_name] = self.train_predictions

        ## the parts below assume that we know the labels...
        if not self.no_labeled_data:
            self._generate_debugging_csv()

            self._generate_rocs()
            self.method_to_optimal_f1_scores_test, self.method_to_optimal_predictions_test, self.method_to_optimal_thresh_test = \
                self._generate_optimal_predictions(self.method_to_test_predictions, self.y_test)

            if not using_pretrained_model:
                self.method_to_optimal_f1_scores_train, self.method_to_optimal_predictions_train, self.method_to_optimal_thresh_train = \
                    self._generate_optimal_predictions(self.method_to_train_predictions, self.y_train)
                self.method_to_cm_df_train = self._generate_confusion_matrixes(self.method_to_optimal_predictions_train, self.y_train,
                                                                    self.exfil_paths_train, self.exfil_weights_train)

            self.method_to_cm_df_test = self._generate_confusion_matrixes(self.method_to_optimal_predictions_test, self.y_test,
                                                                self.exfil_paths_test, self.exfil_weights_test)

            return self.method_to_cm_df_test[self.method_name]
        return None


class single_rate_stats_pipeline():
    def __init__(self, time_gran_to_aggregate_mod_score_dfs, ROC_curve_p, base_output_name, recipes_used,
                 skip_model_part, avg_exfil_per_min, avg_pkt_size, exfil_per_min_variance, pkt_size_variance,
                 no_labeled_data):

        print "STATISTICAL_ANALYSIS_V2"
        self.report_sections = {}
        self.list_of_optimal_fone_scores_at_this_exfil_rates = {}
        self.Xs = {}
        self.Ys = {}
        self.Xts = {}
        self.Yts = {}
        self.no_labeled_data = no_labeled_data
        self.trained_models = {}
        self.timegran_to_methods_to_attacks_found_dfs = {}
        self.timegran_to_methods_toattacks_found_training_df = {}
        self.time_gran_to_outtraffic = {}
        self.timegran_to_statistical_pipeline = {}
        self.base_output_name = base_output_name + '_lasso'
        self.time_gran_to_aggregate_mod_score_dfs = time_gran_to_aggregate_mod_score_dfs
        self.time_gran_to_predicted_test = {}
        self.time_gran_to_cm = {}

        for timegran,feature_df in self.time_gran_to_aggregate_mod_score_dfs.iteritems():
            if timegran not in self.timegran_to_methods_toattacks_found_training_df:
                self.list_of_optimal_fone_scores_at_this_exfil_rates[timegran] = []
                self.Xs[timegran] = []
                self.Ys[timegran] = []
                self.Xts[timegran] = []
                self.Yts[timegran] = []
                self.trained_models[timegran] = []
                self.time_gran_to_outtraffic[timegran] = []
                self.skip_model_part = skip_model_part

        self.recipes_used = recipes_used
        self.avg_exfil_per_min = avg_exfil_per_min
        self.avg_pkt_size = avg_pkt_size
        self.exfil_per_min_variance = exfil_per_min_variance
        self.pkt_size_variance = pkt_size_variance

    def run_statistical_pipeline(self, drop_pairwise_features, pretrained_statistical_analysis_v2, skip_heatmap_p=True):
        for timegran,feature_df in self.time_gran_to_aggregate_mod_score_dfs.iteritems():
            #clf = sklearn.linear_model.LogisticRegressionCV(penalty="l1", cv=10, max_iter=10000, solver='saga')
            #cur_base_output_name + 'logistic_l1_mod_z_lass_feat_sel_'
            clf = LassoCV(cv=3, max_iter=80000)

            if 'lass_feat_sel' in self.base_output_name:
                lasso_feature_selection_p = True
            else:
                lasso_feature_selection_p = False

            stat_pipeline = single_model_stats_pipeline(feature_df, self.base_output_name, self.skip_model_part, clf,
                                                        drop_pairwise_features, timegran, lasso_feature_selection_p,
                                                        skip_heatmap_p=skip_heatmap_p, no_labeled_data=self.no_labeled_data)

            #if pretrained_statistical_analysis_v2 == None or (timegran not in pretrained_statistical_analysis_v2.timegran_to_statistical_pipeline):
            if pretrained_statistical_analysis_v2 == None:
                self.time_gran_to_cm[timegran] = stat_pipeline.generate_model(using_pretrained_model=False)
            else:
                if timegran in pretrained_statistical_analysis_v2.timegran_to_statistical_pipeline:
                    stat_pipeline.clf = pretrained_statistical_analysis_v2.timegran_to_statistical_pipeline[timegran].clf
                    self.time_gran_to_cm[timegran] = stat_pipeline.generate_model(using_pretrained_model=True)

            self.time_gran_to_predicted_test[timegran] = stat_pipeline.test_predictions
            self.list_of_optimal_fone_scores_at_this_exfil_rates[timegran].append(stat_pipeline.method_to_optimal_f1_scores_test)
            self.Xs[timegran].append(stat_pipeline.X_train)
            self.Ys[timegran].append(stat_pipeline.y_train)
            self.Xts[timegran].append(stat_pipeline.X_test)
            self.Yts[timegran].append(stat_pipeline.y_test)
            self.trained_models[timegran].append(stat_pipeline.clf)
            self.timegran_to_statistical_pipeline[timegran] = stat_pipeline
            self.timegran_to_methods_to_attacks_found_dfs[timegran] = stat_pipeline.method_to_cm_df_test
            self.timegran_to_methods_toattacks_found_training_df[timegran] = stat_pipeline.method_to_cm_df_train

            self.time_gran_to_outtraffic[timegran].append(stat_pipeline.out_traffic)

        # if only one time gran, no point in running this function
        if len(self.time_gran_to_aggregate_mod_score_dfs.keys()) > 1:
            self._train_multi_time_model( drop_pairwise_features, pretrained_statistical_analysis_v2, skip_heatmap_p )

    def create_the_report(self, auto_open_p):
            ## handle it like this... self.timegran_to_statistical_pipeline[tuple(self.timegran_to_statistical_pipeline.keys())]
            for timegran,stats_pipeline in self.timegran_to_statistical_pipeline.iteritems():
                self.report_sections[timegran] = stats_pipeline.generate_report_section()

            generate_report.join_report_sections(self.recipes_used, self.base_output_name, self.avg_exfil_per_min,
                                                 self.avg_pkt_size, self.exfil_per_min_variance, self.pkt_size_variance,
                                                 self.report_sections, auto_open_p)

    def find_operating_point_given_fps(self, timegran, fp_limit):
        cur_statspipelines = self.timegran_to_statistical_pipeline[timegran]
        new_cm = cur_statspipelines.find_optimal_cm_given_fps(fp_limit)
        return new_cm

    def generate_return_values(self):

        experiment_info = {}
        experiment_info["recipes_used"] = self.recipes_used
        experiment_info["avg_exfil_per_min"] = self.avg_exfil_per_min
        experiment_info["avg_pkt_size"] = self.avg_pkt_size
        experiment_info["exfil_per_min_variance"] = self.exfil_per_min_variance
        experiment_info["pkt_size_variance"] =self. pkt_size_variance

        return self.list_of_optimal_fone_scores_at_this_exfil_rates, self.Xs, self.Ys, self.Xts, self.Yts, self.trained_models, \
               self.timegran_to_methods_to_attacks_found_dfs, self.timegran_to_methods_toattacks_found_training_df, experiment_info, \
               self.time_gran_to_outtraffic, self.timegran_to_statistical_pipeline


    def _train_multi_time_model(self, drop_pairwise_features, pretrained_statistical_analysis_v2, skip_heatmap_p):
        # the purpose of this function is test the union of alerts...
        ### okay... can I reuse the existing statistical analysis machinery...
        # step 1: get all 0/1 predictions
        timegran_to_predictions = {}
        for time_gran, statspipeline in self.timegran_to_statistical_pipeline.iteritems():
            #test_predictions = list(statspipeline.method_to_test_predictions[statspipeline.method_name])
            #train_predictions = list(statspipeline.method_to_train_predictions[statspipeline.method_name])
            ### why method_to_optimal_predictions_test? -> let's change to just normal predictions...
            #test_predictions = list(statspipeline.method_to_optimal_predictions_test[statspipeline.method_name])
            #train_predictions = list(statspipeline.method_to_optimal_predictions_train[statspipeline.method_name])
            ###
            test_predictions = list(statspipeline.method_to_test_predictions[statspipeline.method_name])
            train_predictions = list(statspipeline.method_to_train_predictions[statspipeline.method_name])
            # method_to_test_predictions
            timegran_to_predictions[time_gran] = train_predictions + test_predictions

        # step 2: take the OR of the predictions
        ## step 2a: convert all other time granularities to largest time granularity
        ## step 2b: take the OR of the elements
        # max_timegran = max( timegran_to_testpredictions.keys() )
        max_timegran = max(timegran_to_predictions.keys())
        final_predictions = {}

        print "timegran_to_predictions", timegran_to_predictions
        for time_gran, predictions in timegran_to_predictions.iteritems():
            # final_trainpredictions[time_gran] = [0 for i in range(0, len(timegran_to_trainpredictions[max_timegran]))]
            # final_testpredictions[time_gran] = [0 for i in range(0, len(timegran_to_testpredictions[max_timegran]))]
            final_predictions[str(time_gran)] = [0 for i in range(0, len(timegran_to_predictions[max_timegran]))]
            conversion_to_max = int(max_timegran / time_gran)  # note: going to assume they all fit in easily
            # for i in range(conversion_to_max,len(testpredictions), conversion_to_max):
            for i in range(0, len(timegran_to_predictions[max_timegran]), 1):
                cur_prediction = sum(predictions[i * conversion_to_max: (i + 1) * conversion_to_max]) / float(
                    conversion_to_max)
                final_predictions[str(time_gran)][i] = final_predictions[str(time_gran)][i] + cur_prediction

        final_predictions = pd.DataFrame(final_predictions, columns=final_predictions.keys())

        ## take the relevant parts from the aggregate mod score and put them up in there too.
        relevant_features_at_timegran = self.timegran_to_statistical_pipeline[max_timegran].aggregate_mod_score_df
        print relevant_features_at_timegran.columns.values
        final_predictions['exfil_path'] = relevant_features_at_timegran['exfil_path']
        final_predictions['concrete_exfil_path'] = relevant_features_at_timegran['concrete_exfil_path']
        final_predictions['exfil_weight'] = relevant_features_at_timegran['exfil_weight']
        final_predictions['exfil_pkts'] = relevant_features_at_timegran['exfil_pkts']
        final_predictions['is_test'] = relevant_features_at_timegran['is_test']
        for column in relevant_features_at_timegran:
            if 'cilium_for_first_sec_' in column:
                final_predictions[column] = relevant_features_at_timegran[column]
                break
        final_predictions['labels'] = relevant_features_at_timegran['labels']
        try:
            final_predictions['real_ide_angles_'] = relevant_features_at_timegran['real_ide_angles_']
        except:
            pass

        timegran = tuple(self.timegran_to_statistical_pipeline.keys())
        # step 3: generate a report (if desired)
        clf = LassoCV(cv=3, max_iter=80000)
        # use the existing machinery in the statistical_pipeline object
        stats_pipeline = single_model_stats_pipeline(final_predictions, self.base_output_name + '_multi_time',
                                                     self.skip_model_part, clf, drop_pairwise_features,
                                                     tuple(self.timegran_to_statistical_pipeline.keys()),
                                                     lasso_feature_selection_p=False, dont_prepare_data_p=True,
                                                     skip_heatmap_p=skip_heatmap_p, no_labeled_data=self.no_labeled_data)

        ### TODO: what is going on here??? it's still fitting of the training data?? but it should on the testing data...
        #if pretrained_statistical_analysis_v2 == None or (timegran not in pretrained_statistical_analysis_v2.timegran_to_statistical_pipeline):
        if pretrained_statistical_analysis_v2 == None:
            self.time_gran_to_cm[timegran] = stats_pipeline.generate_model(using_pretrained_model=False, multi_time_train=True)
            self.time_gran_to_predicted_test[timegran] = stats_pipeline.test_predictions

        else:
            if timegran in pretrained_statistical_analysis_v2.timegran_to_statistical_pipeline:
                stats_pipeline.clf = pretrained_statistical_analysis_v2.timegran_to_statistical_pipeline[timegran].clf
                self.time_gran_to_cm[timegran] = stats_pipeline.generate_model(using_pretrained_model=True, multi_time_train=True)
                self.time_gran_to_predicted_test[timegran] = stats_pipeline.test_predictions

        self.timegran_to_statistical_pipeline[timegran] = stats_pipeline

# not sure if I want to do this... but essentially I could create dict of all the single rate stuff here
## and the use the functions meant to minimize the per-path exfil rates to create the final product.
class statistical_pipeline():
    def __init__(self):
        pass

def drop_useless_columns_aggreg_DF(aggregate_mod_score_dfs):
    # '''
    ## TODO: decide whether to keep / drop -- does not seem to help anything...
    #for column in aggregate_mod_score_dfs:
    #    if ('centrality' in column or 'clustering' in column) and 'class' not in column:
    #        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns=column)

    for column in aggregate_mod_score_dfs:
        #if ('centrality' in column or 'clustering' in column) and 'sub_' not in column:
        #if 'sub_' not in column and column.count('_') >= 3:
        #if 'current_flow_bc_sub_' not in column and column.count('_') >= 3:
        #if 'current_flow_bc_' not in column and column.count('_') >= 3:
        #if 'current_flow_bc_' not in column and column.count('_') >= 3 or 'sub_' in column:
        #if 'current_flow_bc_' not in column and column.count('_') >= 3:

        ##  _reciprocity, _density, _edge_coef_of_var
        #if 'class_harmonic_centrality_' not in column and column.count('_') >= 3 and '_reciprocity' not in column and \
        #    '_edge_coef_of_var' not in column:
        if column.count('_') >= 3 and '_reciprocity' not in column and '_edge_coef_of_var' not in column and \
            'class_harmonic_centrality_' not in column and 'harmonic_centrality_coef_of_var_' not in column and \
                'cilium' not in column and '_ide' not in column:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns=column)
        ##### ### # ### #####

    #amt_of_out_traffic_pkts
    #''' ## put in at some point -- right now it is interesting that it cannot get 100% when the literal
        ## answer is there!
    '''
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='new_neighbors_outsode')  # might wanna just stop these from being generated...
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='new_neighbors_dns')  # might wanna just stop these from being generated...
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='new_neighbors_all')  # might wanna just stop these from being generated...
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='amt_of_out_traffic_bytes')  # might wanna just stop these from being generated...
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='amt_of_out_traffic_pkts')  # might wanna just stop these from being generated...
    except:
        pass
    '''

    '''
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns="Fraction of Communication Between Pods not through VIPs (w abs)_")  # might wanna just stop these from being generated...
    except:
        pass
    '''

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='fraction_pod_comm_but_not_VIP_comms_')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='attack_labels')  # might wanna just stop these from being generated...
    except:
        pass
    #'''

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='pod_1si_density_list_')
        pass
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='new_neighbors_outside')
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='new_neighbors_dns')
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns=u'new_neighbors_all')
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns=u'new_neighbors_all ')
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='fraction_pod_comm_but_not_VIP_comms_no_abs_')
    except:
        pass

    #try:
    #    aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
    #        columns='Communication Between Pods not through VIPs (no abs)_')  # might wanna just stop these from being generated...
    #except:
    #    pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='dns_list_outside_mod_z_score')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='dns_list_inside_mod_z_score')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='into_dns_from_outside_list_mod_z_score')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='into_dns_ratio_mod_z_score')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='kube-dns_to_outside_density_mod_z_score')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='labelsmod_z_score')  # might wanna just stop these from being generated...
    except:
        pass


    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='fraction_pod_comm_but_not_VIP_comms_mod_z_score')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='fraction_pod_comm_but_not_VIP_comms_no_abs_mod_z_score')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='DNS outside_')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='DNS inside_')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='amt_of_out_traffic_bytesmod_z_score')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='amt_of_out_traffic_pktsmod_z_score')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='amt_of_out_traffic_pkts')  # might wanna just stop these from being generated...
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='amt_of_out_traffic_bytes')  # might wanna just stop these from being generated...
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='timemod_z_score')  # might wanna just stop these from being generated...
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='labelsmod_z_score')  # might wanna just stop these from being generaetd
        print aggregate_mod_score_dfs.columns
    except:
        pass
    try:

        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='Unnamed: 0mod_z_score')  # might wanna just stop these from being generaetd
    except:
        pass

    try:

        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='into_dns_from_outside_')  # might wanna just stop these from being generaetd
    except:
        pass

    #try:
    #    aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Communication Between Pods not through VIPs (w abs)_')
    #except:
    #    pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Fraction of Communication Between Pods not through VIPs (no abs)_')
    except:
        pass


    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='pod_comm_but_not_VIP_comms_')
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='pod_comm_but_not_VIP_comms_no_abs_')
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='dns_list_outside_')
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='ide_angles_')
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='dns_list_inside_')
    except:
        pass
    #into_dns_from_outside_list_
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='dns_outside_inside_ratios_')
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='into_dns_from_outside_list_')
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='into_dns_ratio_')
    except:
        pass
    ################
    try:
        #ide_angles
        #aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='ide_angles (w abs)_')
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='ide_angles')
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='DNS outside-to-inside ratio_')
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='into_dns_eigenval_angles_')
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='fraction_pod_comm_but_not_VIP_comms')
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Unnamed: 0')
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='Communication Between Pods not through VIPs (no abs)_mod_z_score')
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='Fraction of Communication Between Pods not through VIPs (no abs)_mod_z_score')
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='DNS inside_mod_z_score')
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='into_dns_from_outside_mod_z_score')
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='DNS outside_mod_z_score')
    except:
        pass
    # '''
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Angle of DNS edge weight vectors_mod_z_score')
    except:
        pass
    # '''
    # '''
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='Angle of DNS edge weight vectors (w abs)_mod_z_score')
    except:
        pass
    # outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio_mod_z_score
    # '''
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio_mod_z_score')
    except:
        pass
    # sum_of_max_pod_to_dns_from_each_svc_mod_z_score
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='sum_of_max_pod_to_dns_from_each_svc_mod_z_score')
    except:
        pass
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='Communication Between Pods not through VIPs (w abs)_')
    except:
        pass

    # Communication Not Through VIPs
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Communication Not Through VIPs')
    except:
        pass
        # '''
        # 'Communication Between Pods not through VIPs (no abs)_mod_z_score'
        # 'Fraction of Communication Between Pods not through VIPs (no abs)_mod_z_score'
    return aggregate_mod_score_dfs

def extract_comparison_methods(X_train, X_test):
    try:
        ewma_train = X_train['max_ewma_control_chart_scores']
        ewma_test = X_test['max_ewma_control_chart_scores']
        X_train = X_train.drop(columns='max_ewma_control_chart_scores')
        X_test = X_test.drop(columns='max_ewma_control_chart_scores')
    except:
        ewma_train = [0 for i in range(0, len(X_train))]
        ewma_test = [0 for i in range(0, len(X_test))]


    #### start complicated ide stuff...
    try:
        ide_train = copy.deepcopy(X_train['real_ide_angles_'])
        copy_of_X_test = X_test.copy(deep=True)
        ide_test = copy.deepcopy(copy_of_X_test['real_ide_angles_'])
        X_train = X_train.drop(columns='real_ide_angles_')
        X_test = X_test.drop(columns='real_ide_angles_')
    except:
        ide_train = [0 for i in range(0, len(X_train))]
        ide_test = [0 for i in range(0, len(X_test))]

    '''
    try:
        # if True:
        print X_train.columns

        try:
            pass
            ide_train = copy.deepcopy(X_train['real_ide_angles_'])
            copy_of_X_test = X_test.copy(deep=True)
            ide_test = copy.deepcopy(copy_of_X_test['real_ide_angles_'])
            X_train = X_train.drop(columns='real_ide_angles_')
            X_test = X_test.drop(columns='real_ide_angles_')
        except:

            ide_train = copy.deepcopy(X_train['ide_angles_w_abs_'])
            #ide_train.fillna(ide_train.mean())
            print "ide_train", ide_train
            # exit(1222)
            copy_of_X_test = X_test.copy(deep=True)
            ide_test = copy.deepcopy(copy_of_X_test['ide_angles_w_abs_'])


            #ide_test = ide_test.fillna(ide_train.mean())
            print "ide_test", ide_test
            X_train = X_train.drop(columns='ide_angles_w_abs_')
            X_test = X_test.drop(columns='ide_angles_w_abs_')
            # if np. ide_test.tolist():
            #    ide_train = [0 for i in range(0, len(X_train))]
            #    ide_test = [0 for i in range(0, len(X_test))]
    except:
        try:
            # ide_train = copy.deepcopy(X_train['ide_angles_mod_z_score'])
            ide_train = copy.deepcopy(X_train['ide_angles (w abs)_mod_z_score'])
            X_train = X_train.drop(columns='ide_angles_mod_z_score')
            X_train = X_train.drop(columns='ide_angles (w abs)_mod_z_score')
            ide_train.fillna(ide_train.mean())
            print "ide_train", ide_train
            # exit(1222)
        except:
            ide_train = [0 for i in range(0, len(X_train))]
        try:
            # copy_of_X_test = X_test.copy(deep=True)
            # ide_test = copy.deepcopy(copy_of_X_test['ide_angles_mod_z_score'])
            ide_test = copy.deepcopy(X_test['ide_angles (w abs)_mod_z_score'])
            X_test = X_test.drop(columns='ide_angles_mod_z_score')
            X_test = X_test.drop(columns='ide_angles (w abs)_mod_z_score')
            ide_test = ide_test.fillna(ide_train.mean())
        except:
            ide_test = [0 for i in range(0, len(X_test))]
    '''
    ########## END complicated ide stuff...

    # cilium_for_first_sec_
    cilium_columns = []
    for column in X_train:
        if 'cilium_for_first_sec_' in column:
            cilium_columns.append(column)
    try:
        cilium_train = copy.deepcopy( X_train[cilium_columns[0]] )
        cilium_test = copy.deepcopy( X_test[cilium_columns[0]] )
        X_train = X_train.drop(columns=cilium_columns[0])
        X_test = X_test.drop(columns=cilium_columns[0])

    except:
        cilium_train = [0 for i in range(0, len(X_train))]
        cilium_test = [0 for i in range(0, len(X_test))]

    return ide_train, ide_test, X_train, X_test, cilium_train, cilium_test

def drop_useless_columns_testTrain_Xs( X_train, X_test ):
    X_train = X_train.drop(columns='exfil_path')
    X_train = X_train.drop(columns='concrete_exfil_path')
    X_train_exfil_weight = X_train['exfil_weight']
    X_train = X_train.drop(columns='exfil_weight')
    X_train = X_train.drop(columns='exfil_pkts')
    X_train = X_train.drop(columns='is_test')
    X_test = X_test.drop(columns='exfil_path')
    X_test = X_test.drop(columns='concrete_exfil_path')
    X_test_exfil_weight = X_test['exfil_weight']
    X_test = X_test.drop(columns='exfil_weight')
    X_test = X_test.drop(columns='exfil_pkts')
    X_test = X_test.drop(columns='is_test')

    try:
        X_test = X_test.drop( columns = 'Communication Between Pods not through VIPs (w abs)' )
    except:
        pass
    try:
        X_train = X_train.drop( columns = 'Communication Between Pods not through VIPs (w abs)' )
    except:
        pass

    print "X_train_columns", X_train.columns, "---"
    try:
        dropped_feature_list = ['New Class-Class Edges with DNS_mod_z_score',
                                'New Class-Class Edges_mod_z_score',
                                'New Class-Class Edges with Outside_mod_z_score',
                                '1-step-induced-pod density_mod_z_score']
        X_train = X_train.drop(columns='New Class-Class Edges with DNS_mod_z_score')
        X_train = X_train.drop(columns='New Class-Class Edges with Outside_mod_z_score')
        X_train = X_train.drop(columns='New Class-Class Edges_mod_z_score')
        X_train = X_train.drop(columns='1-step-induced-pod density_mod_z_score')
        X_test = X_test.drop(columns='New Class-Class Edges with DNS_mod_z_score')
        X_test = X_test.drop(columns='New Class-Class Edges with Outside_mod_z_score')
        X_test = X_test.drop(columns='New Class-Class Edges_mod_z_score')
        X_test = X_test.drop(columns='1-step-induced-pod density_mod_z_score')

    except:
        dropped_feature_list = ['New Class-Class Edges with DNS_', 'New Class-Class Edges with Outside_',
                                'New Class-Class Edges_',
                                '1-step-induced-pod density_']
        print "X_train_columns", X_train.columns, "---"
        try:
            X_train = X_train.drop(columns='New Class-Class Edges with DNS_')
        except:
            pass
        try:
            X_train = X_train.drop(columns='New Class-Class Edges with Outside_')
        except:
            pass
        try:
            X_train = X_train.drop(columns='New Class-Class Edges_')
        except:
            pass
        try:
            X_train = X_train.drop(columns='1-step-induced-pod density_')
        except:
            pass
        try:
            X_test = X_test.drop(columns='New Class-Class Edges with DNS_')
        except:
            pass
        try:
            X_test = X_test.drop(columns='New Class-Class Edges with Outside_')
        except:
            pass
        try:
            X_test = X_test.drop(columns='New Class-Class Edges_')
        except:
            pass
        try:
            X_test = X_test.drop(columns='1-step-induced-pod density_')
        except:
            pass

    return X_train, X_test, dropped_feature_list, X_train_exfil_weight, X_test_exfil_weight

def get_coef_dict(clf, X_train_columns, base_output_name, X_train_dtypes, sanity_check_number_coefs):
    coef_dict = {}
    print "Coefficients: "
    print "LASSO model", clf.get_params()
    print '----------------------'
    print "len(clf.coef_)", len(clf.coef_), "len(X_train_columns)", len(X_train_columns)

    if 'logistic' in base_output_name:
        model_coefs = clf.coef_[0]
    else:
        model_coefs = clf.coef_

    if sanity_check_number_coefs:
        if len(model_coefs) != (len(X_train_columns)):  # there is no plus one b/c the intercept is stored in clf.intercept_
            print "coef_ is different length than X_train_columns!", X_train_columns
            for counter, i in enumerate(X_train_dtypes):
                print counter, i, X_train_columns[counter]
                print model_coefs  # [counter]
                print len(model_coefs)
            exit(888)

    for coef, feat in zip(model_coefs, X_train_columns):
        coef_dict[feat] = coef
    coef_dict['intercept'] = float(clf.intercept_)

    print "COEFS_HERE"
    for coef, feature in coef_dict.iteritems():
        print coef, feature

    print "coef_dict"
    return coef_dict

def prepare_data(aggregate_mod_score_dfs, skip_model_part, time_gran_to_debugging_csv, time_gran,
                 drop_pairwise_features):

    out_traffic=None
    '''
    try:
        out_traffic = aggregate_mod_score_dfs['amt_of_out_traffic_bytes']
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='amt_of_out_traffic_bytes')  # might wanna just stop these from being generated...
    except:
        pass
    '''

    aggregate_mod_score_dfs = drop_useless_columns_aggreg_DF(aggregate_mod_score_dfs)

    if drop_pairwise_features:
        aggregate_mod_score_dfs = drop_pairwise_features_func(aggregate_mod_score_dfs)

    #if not skip_model_part:
    #if ignore_physical_attacks_p:
    #    aggregate_mod_score_dfs = \
     #       aggregate_mod_score_dfs[~((aggregate_mod_score_dfs['labels'] == 1) &
     #                                 ((aggregate_mod_score_dfs['exfil_pkts'] == 0) &
     #                                  (aggregate_mod_score_dfs['exfil_weight'] == 0)))]

    aggregate_mod_score_dfs_training = aggregate_mod_score_dfs[aggregate_mod_score_dfs['is_test'] == 0]
    aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs[aggregate_mod_score_dfs['is_test'] == 1]
    time_gran_to_debugging_csv[time_gran] = aggregate_mod_score_dfs.copy(deep=True)
    print "aggregate_mod_score_dfs_training", aggregate_mod_score_dfs_training
    print "aggregate_mod_score_dfs_testing", aggregate_mod_score_dfs_testing
    print aggregate_mod_score_dfs['is_test']

    #else:
        ## note: generally you'd want to split into test and train sets, but if we're not doing logic
        ## part anyway, we just want quick-and-dirty results, so don't bother (note: so for formal purposes,
        ## DO NOT USE WITHOUT LOGIC CHECKING OR SOLVE THE TRAINING-TESTING split problem)
    #    aggregate_mod_score_dfs_training, aggregate_mod_score_dfs_testing = train_test_split(aggregate_mod_score_dfs,
    #                                                                                         test_size=0.5)
    #    time_gran_to_debugging_csv[time_gran] = aggregate_mod_score_dfs_training.copy(deep=True).append(
    #        aggregate_mod_score_dfs_testing.copy(deep=True))

    print aggregate_mod_score_dfs_training.index
    #aggregate_mod_score_dfs_training, aggregate_mod_score_dfs_testing = \
    #    drop_useless_columns_aggreg_testtrain_DF(aggregate_mod_score_dfs_training, aggregate_mod_score_dfs_testing)

    X_train = aggregate_mod_score_dfs_training.loc[:, aggregate_mod_score_dfs_training.columns != 'labels']
    y_train = aggregate_mod_score_dfs_training.loc[:, aggregate_mod_score_dfs_training.columns == 'labels']
    X_test = aggregate_mod_score_dfs_testing.loc[:, aggregate_mod_score_dfs_training.columns != 'labels']
    y_test = aggregate_mod_score_dfs_testing.loc[:, aggregate_mod_score_dfs_training.columns == 'labels']

    # get method to compare against and remove them from the DF...
    ide_train, ide_test, X_train, X_test, cilium_train, cilium_test = extract_comparison_methods(X_train, X_test)

    exfil_paths = X_test['exfil_path']
    exfil_paths_train = X_train['exfil_path']
    try:
        exfil_paths = exfil_paths.replace('0', '[]')
        exfil_paths_train = exfil_paths_train.replace('0', '[]')
    except:
        try:
            exfil_paths = exfil_paths.replace(0, '[]')
            exfil_paths_train = exfil_paths_train.replace(0, '[]')
        except:
            pass

    X_train, X_test, dropped_feature_list, X_train_exfil_weight, X_test_exfil_weight = \
        drop_useless_columns_testTrain_Xs(X_train, X_test)


    print '-------'
    print type(X_train)
    print "X_train_columns_values", X_train.columns.values
    print "columns", X_train.columns
    print "columns", X_test.columns

    print X_train.dtypes

    # need to replace the missing values in the data w/ meaningful values...
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    print "X_train_median", X_train.median()

    print X_train
    pre_drop_X_train = X_train.copy(deep=True)
    X_train = X_train.dropna(axis=1)
    print X_train
    X_test = X_test.dropna(axis=1)

    return X_train, y_train, X_test, y_test, pre_drop_X_train, time_gran_to_debugging_csv, dropped_feature_list, \
           ide_train, ide_test, X_train_exfil_weight, X_test_exfil_weight, exfil_paths, exfil_paths_train, out_traffic,\
            cilium_train, cilium_test


def lasso_feature_selection(X_train, y_train, X_test, y_test):
    clf_featuree_selection = LassoCV(cv=5)
    sfm = sklearn.feature_selection.SelectFromModel(clf_featuree_selection)
    sfm.fit(X_train, y_train)
    feature_idx = sfm.get_support()
    selected_columns = X_train.columns[feature_idx]
    X_train = pd.DataFrame(sfm.transform(X_train), index=X_train.index, columns=selected_columns)
    X_test = pd.DataFrame(sfm.transform(X_test), index=X_test.index, columns=selected_columns)
    # X_test = sfm.transform(X_test)
    return X_train, X_test

def drop_pairwise_features_func(aggregate_mod_score_dfs):

    for column in aggregate_mod_score_dfs:
        try:
            if '_to_' in column:
                aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns=column)
        except:
            pass
    return aggregate_mod_score_dfs

def generate_ROC_curves(y_test, test_predictions, base_output_name, time_gran, ide_test, ide_train,
                        list_of_optimal_fone_scores, cilium_train, cilium_test):
    ## use the generate sklearn model to create the detection ROC
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=y_test, y_score=test_predictions, pos_label=1)
    x_vals = fpr
    y_vals = tpr
    ROC_path = base_output_name + '_good_roc_'
    title = 'ROC Linear Combination of Features at ' + str(time_gran)
    plot_name = 'sub_roc_lin_comb_features_' + str(time_gran)

    try:
        os.makedirs('./temp_outputs')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    ##  ewma_train = X_train['max_ewma_control_chart_scores']
    ##  ewma_test = X_test['max_ewma_control_chart_scores']
    # now for the ewma part...
    # fpr_ewma, tpr_ewma, thresholds_ewma = sklearn.metrics.roc_curve(y_true=y_test, y_score = ewma_test, pos_label=1)
    print "y_test", y_test
    print "ide_test", ide_test, ide_train
    try:
        fpr_ide, tpr_ide, thresholds_ide = sklearn.metrics.roc_curve(y_true=y_test, y_score=ide_test, pos_label=1)
        line_titles = ['ensemble model', 'ide_angles']
        list_of_x_vals = [x_vals, fpr_ide]
        list_of_y_vals = [y_vals, tpr_ide]
    except:
        # ide_test = [0 for i in range(0, len(X_test))]
        # fpr_ide, tpr_ide, thresholds_ide = sklearn.metrics.roc_curve(y_true=y_test, y_score = ide_test, pos_label=1)
        line_titles = ['ensemble model']
        list_of_x_vals = [x_vals]
        list_of_y_vals = [y_vals]

    try:
        fpr_cil, tpr_cil, thresholds_cil = sklearn.metrics.roc_curve(y_true=y_test, y_score=cilium_test, pos_label=1)
        line_titles.append('cilium')
        list_of_x_vals.append(fpr_cil)
        list_of_y_vals.append(tpr_cil)
    except:
        pass

    ax, _, plot_path = construct_ROC_curve(list_of_x_vals, list_of_y_vals, title, ROC_path + plot_name, \
                                           line_titles, show_p=False)

    ### determination of the optimal operating point goes here (take all the thresh vals and predictions,
    ### find the corresponding f1 scores (using sklearn func), and then return the best.
    optimal_f1_score, optimal_thresh = process_roc.determine_optimal_threshold(y_test, test_predictions, thresholds)
    print "optimal_f1_score", optimal_f1_score, "optimal_thresh", optimal_thresh
    list_of_optimal_fone_scores.append(optimal_f1_score)
    ### get confusion matrix... take predictions from classifer. THreshold
    ### using optimal threshold determined previously. Extract the labels too. This gives two lists, appropriate
    ### for using the confusion_matrix function of sklearn. However, this does NOT handle the different
    ### categories... (for display will probably want to make a df)
    optimal_predictions = [int(i > optimal_thresh) for i in test_predictions]
    print "optimal_predictions", optimal_predictions
    ### determine categorical-level behavior... Split the two lists from the previous step into 2N lists,
    ### where N is the # of categories, and then can just do the confusion matrix function on them...
    ### (and then display the results somehow...)

    optimal_f1_score_ide, ide_optimal_thresh = process_roc.determine_optimal_threshold(y_test, ide_test, thresholds_ide)

    return optimal_predictions, optimal_thresh, plot_path, ide_optimal_thresh


def determine_categorical_cm_df(y_test, optimal_predictions, exfil_paths, exfil_weights):
    y_test = y_test['labels'].tolist()
    print "new_y_test", y_test
    attack_type_to_predictions, attack_type_to_truth, attack_type_to_weights = \
        process_roc.determine_categorical_labels(y_test, optimal_predictions, exfil_paths, exfil_weights.tolist())
    attack_type_to_confusion_matrix_values = process_roc.determine_cm_vals_for_categories(attack_type_to_predictions,
                                                                                          attack_type_to_truth)
    categorical_cm_df = process_roc.determine_categorical_cm_df(attack_type_to_confusion_matrix_values, attack_type_to_weights)
    ## re-name the row without any attacks in it...
    print "categorical_cm_df.index", categorical_cm_df.index
    categorical_cm_df = categorical_cm_df.rename({(): 'No Attack'}, axis='index')
    return categorical_cm_df


def construct_ROC_curve(list_of_x_vals, list_of_y_vals, title, plot_name, line_titles, show_p=False):
    # okay, what we want to do here is to construct
    # x_vals should be FPR
    # y_vals should be TPR
    #plt.figure()
    fig, axs = plt.subplots(1,1)
    plt.ylim(-0.05,1.05)
    plt.xlim(-0.05,1.05)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)

    line_markers = ['s', '*', 'h', '+', '1']
    print "list_of_x_vals", list_of_x_vals
    print "list_of_y_vals", list_of_y_vals
    print "line_titles", line_titles
    for counter,x_vals in enumerate(list_of_x_vals):
        plt.plot(x_vals, list_of_y_vals[counter], label = line_titles[counter], marker=line_markers[counter])

    plt.legend()
    plt.savefig( plot_name + '.png', format='png', dpi=1000)

    local_graph_loc = './temp_outputs/' + title + '.png'
    local_graph_loc = local_graph_loc.replace(' ', '_')
    plt.savefig( local_graph_loc, format='png', dpi=1000)

    if show_p:
        plt.show()
    plt.close()
    return axs, plot_name + '.png', '../' + local_graph_loc[2:]