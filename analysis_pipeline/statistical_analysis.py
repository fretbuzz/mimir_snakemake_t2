import copy
import errno
import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

from analysis_pipeline import generate_heatmap, generate_alerts, process_roc, generate_report


def statistically_analyze_graph_features(time_gran_to_aggregate_mod_score_dfs, ROC_curve_p, base_output_name, names,
                                         starts_of_testing, path_occurence_training_df, path_occurence_testing_df,
                                         recipes_used, skip_model_part, clf, ignore_physical_attacks_p,
                                         fraction_of_edge_weights, fraction_of_edge_pkts):
    #print time_gran_to_aggregate_mod_score_dfs['60']
    ######### 2a.II. do the actual splitting
    # note: labels have the column name 'labels' (noice)
    time_gran_to_model = {}
    #images = 0
    time_gran_to_debugging_csv = {} # note: going to be used for (shockingly) debugging purposes....
    percent_attacks = []
    list_percent_attacks_training = []
    list_of_rocs = []
    list_of_feat_coefs_dfs = []
    time_grans = []
    list_of_model_parameters = []
    list_of_attacks_found_dfs = []
    list_of_attacks_found_training_df = []
    list_of_optimal_fone_scores = []
    feature_activation_heatmaps = []
    feature_raw_heatmaps = []
    ideal_thresholds = []
    feature_activation_heatmaps_training, feature_raw_heatmaps_training = [], []
    X_trains = {}
    Y_trains = {}
    X_tests = {}
    Y_tests = {}
    trained_models = {}
    for time_gran,aggregate_mod_score_dfs in time_gran_to_aggregate_mod_score_dfs.iteritems():
        # drop columns with all identical values b/c they are useless and too many of them makes LASSO wierd
        #aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(aggregate_mod_score_dfs.std()[(aggregate_mod_score_dfs.std() == 0)].index, axis=1)

        '''
        print aggregate_mod_score_dfs.columns
        for column in aggregate_mod_score_dfs.columns:
            if 'coef_of_var_' in column or 'reciprocity' in column or '_density_' in column: # todo
                aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(column, axis=1)
        '''

        time_grans.append(time_gran)
        #'''
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='timemod_z_score')   # might wanna just stop these from being generated...
        except:
            pass
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='labelsmod_z_score') # might wanna just stop these from being generaetd
            print aggregate_mod_score_dfs.columns
        except:
            pass
        try:

            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Unnamed: 0mod_z_score') # might wanna just stop these from being generaetd
        except:
            pass

        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Unnamed: 0')
        except:
            pass
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Communication Between Pods not through VIPs (no abs)_mod_z_score')
        except:
            pass
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Fraction of Communication Between Pods not through VIPs (no abs)_mod_z_score')
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
        #'''
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Angle of DNS edge weight vectors_mod_z_score')
        except:
            pass
        #'''
        #'''
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Angle of DNS edge weight vectors (w abs)_mod_z_score')
        except:
            pass
        #outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio_mod_z_score
        #'''
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='outside_to_sum_of_max_pod_to_dns_from_each_svc_ratio_mod_z_score')
        except:
            pass
        #sum_of_max_pod_to_dns_from_each_svc_mod_z_score
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='sum_of_max_pod_to_dns_from_each_svc_mod_z_score')
        except:
            pass
        #Communication Not Through VIPs
        try:
            aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Communication Not Through VIPs')
        except:
            pass
        #'''
        #'Communication Between Pods not through VIPs (no abs)_mod_z_score'
        #'Fraction of Communication Between Pods not through VIPs (no abs)_mod_z_score'

        #'''
        if not skip_model_part:
            ### TODO TODO TODO TODO TODO TODO
            ### todo: might wanna remove? might wanna keep? not sure...
            ### todo: drop the test physical attacks from the test sets...
            #'''
            if ignore_physical_attacks_p:
                aggregate_mod_score_dfs = \
                aggregate_mod_score_dfs[~((aggregate_mod_score_dfs['labels'] == 1) &
                                          ((aggregate_mod_score_dfs['exfil_pkts'] == 0) &
                                           (aggregate_mod_score_dfs['exfil_weight'] == 0)) )]
            #'''
            #####

            aggregate_mod_score_dfs_training = aggregate_mod_score_dfs[aggregate_mod_score_dfs['is_test'] == 0]
            aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs[aggregate_mod_score_dfs['is_test'] == 1]
            time_gran_to_debugging_csv[time_gran] = aggregate_mod_score_dfs.copy(deep=True)
            print "aggregate_mod_score_dfs_training",aggregate_mod_score_dfs_training
            print "aggregate_mod_score_dfs_testing",aggregate_mod_score_dfs_testing
            print aggregate_mod_score_dfs['is_test']
            #exit(344)

        else:
            ## note: generally you'd want to split into test and train sets, but if we're not doing logic
            ## part anyway, we just want quick-and-dirty results, so don't bother (note: so for formal purposes,
            ## DO NOT USE WITHOUT LOGIC CHECKING OR SOLVE THE TRAINING-TESTING split problem)
            aggregate_mod_score_dfs_training, aggregate_mod_score_dfs_testing = train_test_split(aggregate_mod_score_dfs, test_size=0.5)
            #aggregate_mod_score_dfs_training = aggregate_mod_score_dfs
            #aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs
            time_gran_to_debugging_csv[time_gran] = aggregate_mod_score_dfs_training.copy(deep=True).append(aggregate_mod_score_dfs_testing.copy(deep=True))

        #time_gran_to_debugging_csv[time_gran] = copy.deepcopy(aggregate_mod_score_dfs)

        print aggregate_mod_score_dfs_training.index
        aggregate_mod_score_dfs_training = aggregate_mod_score_dfs_training.drop(columns='new_neighbors_outside')
        aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs_testing.drop(columns='new_neighbors_outside')
        aggregate_mod_score_dfs_training = aggregate_mod_score_dfs_training.drop(columns='new_neighbors_dns')
        aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs_testing.drop(columns='new_neighbors_dns')
        try:
            aggregate_mod_score_dfs_training = aggregate_mod_score_dfs_training.drop(columns=u'new_neighbors_all')
            aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs_testing.drop(columns=u'new_neighbors_all')
        except:
            pass
        try:
            aggregate_mod_score_dfs_training = aggregate_mod_score_dfs_training.drop(columns=u'new_neighbors_all ')
            aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs_testing.drop(columns=u'new_neighbors_all ')
        except:
            pass

        X_train = aggregate_mod_score_dfs_training.loc[:, aggregate_mod_score_dfs_training.columns != 'labels']
        y_train = aggregate_mod_score_dfs_training.loc[:, aggregate_mod_score_dfs_training.columns == 'labels']
        X_test = aggregate_mod_score_dfs_testing.loc[:, aggregate_mod_score_dfs_training.columns != 'labels']
        y_test = aggregate_mod_score_dfs_testing.loc[:, aggregate_mod_score_dfs_training.columns == 'labels']

        #print "X_train", X_train
        #print "y_train", y_train

        ##X_train, X_test, y_train, y_test =  sklearn.model_selection.train_test_split(X, y, test_size = 1-goal_train_test_split, random_state = 42)
        #print X_train.shape, "X_train.shape"

        exfil_paths = X_test['exfil_path'].replace('0','[]')
        exfil_paths_train = X_train['exfil_path'].replace('0','[]')
        #print "----"
        #print "exfil_path_pre_literal_eval", exfil_paths, type(exfil_paths)
        #exfil_paths = ast.literal_eval(exfil_paths)
        #print "----"

        ## todo: extract and put in a 'safe' spot...
        ##calculated_values['max_ewma_control_chart_scores'] = list_of_max_ewma_control_chart_scores
        try:
            ewma_train = X_train['max_ewma_control_chart_scores']
            ewma_test = X_test['max_ewma_control_chart_scores']
            X_train = X_train.drop(columns='max_ewma_control_chart_scores')
            X_test = X_test.drop(columns='max_ewma_control_chart_scores')
        except:
            ewma_train = [0 for i in range(0,len(X_train))]
            ewma_test = [0 for i in range(0,len(X_test))]

        try:
            #if True:
            print X_train.columns
            ide_train =  copy.deepcopy(X_train['ide_angles_'])
            ide_train.fillna(ide_train.mean())
            print "ide_train", ide_train
            #exit(1222)
            copy_of_X_test = X_test.copy(deep=True)
            ide_test = copy.deepcopy(copy_of_X_test['ide_angles_'])
            ide_test = ide_test.fillna(ide_train.mean())
            print "ide_test",ide_test
            X_train = X_train.drop(columns='ide_angles_')
            X_test = X_test.drop(columns='ide_angles_')
            #if np. ide_test.tolist():
            #    ide_train = [0 for i in range(0, len(X_train))]
            #    ide_test = [0 for i in range(0, len(X_test))]
        except:
            try:
                #ide_train = copy.deepcopy(X_train['ide_angles_mod_z_score'])
                ide_train = copy.deepcopy(X_train['ide_angles (w abs)_mod_z_score'])
                X_train = X_train.drop(columns='ide_angles_mod_z_score')
                X_train = X_train.drop(columns='ide_angles (w abs)_mod_z_score')
                ide_train.fillna(ide_train.mean())
                print "ide_train", ide_train
                # exit(1222)
            except:
                ide_train = [0 for i in range(0,len(X_train))]
            try:
                #copy_of_X_test = X_test.copy(deep=True)
                #ide_test = copy.deepcopy(copy_of_X_test['ide_angles_mod_z_score'])
                ide_test = copy.deepcopy(X_test['ide_angles (w abs)_mod_z_score'])
                X_test = X_test.drop(columns='ide_angles_mod_z_score')
                X_test = X_test.drop(columns='ide_angles (w abs)_mod_z_score')
                ide_test = ide_test.fillna(ide_train.mean())
            except:
                ide_test = [0 for i in range(0,len(X_test))]


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

        ## TODO: might to put these back in...
        dropped_feature_list = []

        #'''
        print "X_train_columns", X_train.columns, "---"
        try:
            ## TODO: probably wanna keep the outside_mod_z_score in...
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
            print "X_train_columns",X_train.columns, "---"
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
        #'''

        print '-------'
        print type(X_train)
        print "X_train_columns_values", X_train.columns.values
        ###exit(344) ### TODO TODO TODO <<<----- remove!!!

        print "columns", X_train.columns
        print "columns", X_test.columns

        print X_train.dtypes
        # need to replace the missing values in the data w/ meaningful values...
        ''' ## imputer is dropping a column... let's do this w/ pandas dataframes instead....
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp = imp.fit(X_train)
        X_train = imp.transform(X_train)
        X_test = imp.transform(X_test)
        '''
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        print "X_train_median", X_train.median()

        print X_train
        #exit(233)
        pre_drop_X_train = X_train.copy(deep=True)
        X_train = X_train.dropna(axis=1)
        print X_train
        #exit(233)
        X_test = X_test.dropna(axis=1)

        ## TODO: okay, let's try to use the lasso for feature selection by logistic regresesion for the actual model...
        ''' ## TODO: seperate feature selection step goes here...
        clf_featuree_selection = LassoCV(cv=5)
        # Set a minimum threshold of 0.25
        sfm = sklearn.SelectFromModel(clf_featuree_selection)
        sfm.fit(X, y)
        n_features = sfm.transform(X).shape[1]
        '''
        if 'lass_feat_sel' in base_output_name:
            clf_featuree_selection = LassoCV(cv=5)
            sfm = sklearn.feature_selection.SelectFromModel(clf_featuree_selection)
            sfm.fit(X_train, y_train)
            feature_idx = sfm.get_support()
            selected_columns = X_train.columns[feature_idx]
            X_train = pd.DataFrame(sfm.transform(X_train),index=X_train.index,columns=selected_columns)
            X_test = pd.DataFrame(sfm.transform(X_test),index=X_test.index,columns=selected_columns)
            #X_test = sfm.transform(X_test)


        dropped_columns = list(pre_drop_X_train.columns.difference(X_train.columns))
        print "dropped_columns", dropped_columns
        #exit(233)
        #dropped_columns=[]

        print "columns", X_train.columns
        print "columns", X_test.columns

        X_train_dtypes = X_train.dtypes
        X_train_columns = X_train.columns.values
        X_test_columns = X_test.columns.values
        print y_test
        number_attacks_in_test = len(y_test[y_test['labels'] == 1])
        number_non_attacks_in_test = len(y_test[y_test['labels'] == 0])
        percent_attacks.append(float(number_attacks_in_test) / (number_non_attacks_in_test + number_attacks_in_test))

        print y_train
        number_attacks_in_train = len(y_train[y_train['labels'] == 1])
        number_non_attacks_in_train = len(y_train[y_train['labels'] == 0])
        print number_non_attacks_in_train,number_attacks_in_train
        list_percent_attacks_training.append(float(number_attacks_in_train) / (number_non_attacks_in_train + number_attacks_in_train))

        #print "X_train", X_train
        #print "y_train", y_train, len(y_train)
        #print "y_test", y_test, len(y_test)
        #print "-- y_train", len(y_train), "y_test", len(y_test), "time_gran", time_gran, "--"


        ### train the model and generate predictions (2B)
        # note: I think this is where I'd need to modify it to make the anomaly-detection using edge correlation work...

        #clf = sklearn.tree.DecisionTreeClassifier()
        #clf = RandomForestClassifier(n_estimators=10)
        #clf = clf.fit(X_train, y_train)

        #clf = ElasticNetCV(l1_ratio=1.0)
        #clf = RidgeCV(cv=10) ## TODO TODO TODO <<-- instead of having choosing the alpha be magic, let's use cross validation to choose it instead...
        #alpha = 5 # note: not used unless the line underneath is un-commented...
        #clf=Lasso(alpha=alpha)
        print X_train.dtypes
        print y_train

        clf.fit(X_train, y_train)
        trained_models[time_gran] = clf
        score_val = clf.score(X_test, y_test)
        print "score_val", score_val
        test_predictions = clf.predict(X=X_test)
        train_predictions = clf.predict(X=X_train)
        #coefficients = pd.DataFrame({"Feature": X.columns, "Coefficients": np.transpose(clf.coef_)})
        #print coefficients
        #clf.coef_, "intercept", clf.intercept_

        coef_dict = {}
        #coef_feature_df = pd.DataFrame() # TODO: remove if we go back to LASSO
        #'''
        ### get the coefficients used in the model...
        print "Coefficients: "
        print "LASSO model", clf.get_params()
        print '----------------------'
        print len(time_gran_to_debugging_csv[time_gran]["labels"]), len(np.concatenate([train_predictions, test_predictions]))
        print len(time_gran_to_debugging_csv[time_gran].index)
        if not skip_model_part:
            time_gran_to_debugging_csv[time_gran].loc[:, "aggreg_anom_score"] = np.concatenate([train_predictions, test_predictions])
        else:
            #time_gran_to_debugging_csv[time_gran].loc[:, "aggreg_anom_score"] = test_predictions
            time_gran_to_debugging_csv[time_gran].loc[:, "aggreg_anom_score"] = np.concatenate([train_predictions, test_predictions])

        print "len(clf.coef_)", len(clf.coef_), "len(X_train_columns)", len(X_train_columns), "time_gran", time_gran, \
            "len(X_test_columns)", len(X_test_columns), X_train.shape, X_test.shape

        if 'logistic' in base_output_name:
            model_coefs = clf.coef_[0]
        else:
            model_coefs = clf.coef_

        if len(model_coefs) != (len(X_train_columns)): # there is no plus one b/c the intercept is stored in clf.intercept_
            print "coef_ is different length than X_train_columns!", X_train_columns
            for  counter,i in enumerate(X_train_dtypes):
                print counter,i, X_train_columns[counter]
                print model_coefs#[counter]
                print len(model_coefs)
            exit(888)
        for coef, feat in zip(model_coefs, X_train_columns):
            coef_dict[feat] = coef
        print "COEFS_HERE"
        print "intercept...", clf.intercept_
        coef_dict['intercept'] = clf.intercept_
        for coef,feature in coef_dict.iteritems():
            print coef,feature
        #exit(233) ## TODO REMOVE!!!!

        #print "COEF_DICT", coef_dict

        coef_feature_df = pd.DataFrame.from_dict(coef_dict, orient='index')

        #plt.savefig(local_graph_loc, format='png', dpi=1000)
        #print coef_feature_df.columns.values
        #coef_feature_df.index.name = 'Features'
        coef_feature_df.columns = ['Coefficient']
        #'''


        print '--------------------------'

        model_params = clf.get_params()
        try:
            model_params['alpha_val'] = clf.alpha_
        except:
            try:
                pass
                #model_params['alpha_val'] = alpha
            except:
                pass
                #tree.export_graphviz(clf,out_file = 'tree.dot')
                #pass
        list_of_model_parameters.append(model_params)
        '''
        print X_train
        coefficients = clf.coef_
        print len(coefficients), "<- len coefficients,", len(list(X.columns.values))
        for counter,column in enumerate(list(X.columns.values)):
            print counter
            print column, coefficients[counter]
        print "score_val", score_val
        time_gran_to_model[time_gran] = clf
        ##print "time_gran", time_gran, "scores", scores
        '''

        current_heatmap_val_path = base_output_name + 'coef_val_heatmap_' + str(time_gran) + '.png'
        local_heatmap_val_path = 'temp_outputs/heatmap_coef_val_at_' +  str(time_gran) + '.png'
        current_heatmap_path = base_output_name + 'coef_act_heatmap_' + str(time_gran) + '.png'
        local_heatmap_path = 'temp_outputs/heatmap_coef_contribs_at_' +  str(time_gran) + '.png'
        coef_impact_df, raw_feature_val_df = generate_heatmap.generate_covariate_heatmap(coef_dict, X_test, exfil_paths)
        generate_heatmap.generate_heatmap(coef_impact_df, local_heatmap_path, current_heatmap_path)
        generate_heatmap.generate_heatmap(raw_feature_val_df, local_heatmap_val_path, current_heatmap_val_path)
        feature_activation_heatmaps.append('../' + local_heatmap_path)
        feature_raw_heatmaps.append('../' + local_heatmap_val_path)
        print coef_impact_df
        #exit(233)

        current_heatmap_raw_val_path_training = base_output_name + 'training_raw_val_heatmap_' + str(time_gran) + '.png'
        local_heatmap_raw_val_path_training = 'temp_outputs/training_heatmap_raw_val_at_' +  str(time_gran) + '.png'
        current_heatmap_path_training = base_output_name + 'training_coef_act_heatmap_' + str(time_gran) + '.png'
        local_heatmap_path_training = 'temp_outputs/training_heatmap_coef_contribs_at_' +  str(time_gran) + '.png'
        coef_impact_df, raw_feature_val_df = generate_heatmap.generate_covariate_heatmap(coef_dict, X_train, exfil_paths_train)
        generate_heatmap.generate_heatmap(coef_impact_df, local_heatmap_path_training, current_heatmap_path_training)
        generate_heatmap.generate_heatmap(raw_feature_val_df, local_heatmap_raw_val_path_training, current_heatmap_raw_val_path_training)
        feature_activation_heatmaps_training.append('../' + local_heatmap_path_training)
        feature_raw_heatmaps_training.append('../' + local_heatmap_raw_val_path_training)

        X_trains[time_gran] = X_train
        Y_trains[time_gran] = y_train
        X_tests[time_gran] = X_test
        Y_tests[time_gran] = y_test

        ### step (3)
        ## use the generate sklearn model to create the detection ROC
        if ROC_curve_p:
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
            #fpr_ewma, tpr_ewma, thresholds_ewma = sklearn.metrics.roc_curve(y_true=y_test, y_score = ewma_test, pos_label=1)
            print "y_test",y_test
            print "ide_test",ide_test, ide_train
            try:
                fpr_ide, tpr_ide, thresholds_ide = sklearn.metrics.roc_curve(y_true=y_test, y_score = ide_test, pos_label=1)
                line_titles = ['ensemble model', 'ide_angles']
                list_of_x_vals = [x_vals, fpr_ide]
                list_of_y_vals = [y_vals, tpr_ide]
            except:
                #ide_test = [0 for i in range(0, len(X_test))]
                #fpr_ide, tpr_ide, thresholds_ide = sklearn.metrics.roc_curve(y_true=y_test, y_score = ide_test, pos_label=1)
                line_titles = ['ensemble model']
                list_of_x_vals = [x_vals]
                list_of_y_vals = [y_vals]

            ax, _, plot_path = generate_alerts.construct_ROC_curve(list_of_x_vals, list_of_y_vals, title, ROC_path + plot_name,\
                                                                   line_titles, show_p=False)
            list_of_rocs.append(plot_path)

            ### determination of the optimal operating point goes here (take all the thresh vals and predictions,
            ### find the corresponding f1 scores (using sklearn func), and then return the best.
            optimal_f1_score, optimal_thresh = process_roc.determine_optimal_threshold(y_test, test_predictions, thresholds)
            ideal_thresholds.append(optimal_thresh)
            print "optimal_f1_score", optimal_f1_score, "optimal_thresh",optimal_thresh
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

            categorical_cm_df = determine_categorical_cm_df(y_test, optimal_predictions, exfil_paths, X_test_exfil_weight)
            list_of_attacks_found_dfs.append(categorical_cm_df)

            optimal_train_predictions = [int(i>optimal_thresh) for i in train_predictions]
            categorical_cm_df_training = determine_categorical_cm_df(y_train, optimal_train_predictions, exfil_paths_train,
                                                                     X_train_exfil_weight)
            list_of_attacks_found_training_df.append(categorical_cm_df_training)

            if not skip_model_part:
                time_gran_to_debugging_csv[time_gran].loc[:, "anom_val_at_opt_pt"] = \
                    np.concatenate([optimal_train_predictions, optimal_predictions])
            else:
                #time_gran_to_debugging_csv[time_gran].loc[:, "anom_val_at_opt_pt"] = optimal_predictions
                time_gran_to_debugging_csv[time_gran].loc[:, "anom_val_at_opt_pt"] = \
                    np.concatenate([optimal_train_predictions, optimal_predictions])

            # I don't want the attributes w/ zero coefficients to show up in the debugging csv b/c it makes it hard to read
            ## TODO
            for feature,coef in coef_dict.iteritems():
                print "coef_check", coef, not coef, feature
                if not coef:
                    print "just_dropped", feature
                    try:
                        time_gran_to_debugging_csv[time_gran] = time_gran_to_debugging_csv[time_gran].drop([feature],axis=1)
                        coef_feature_df = coef_feature_df.drop(feature, axis=0)
                    except:
                        pass
                for dropped_feature in dropped_feature_list + dropped_columns:
                    try:
                        time_gran_to_debugging_csv[time_gran] = time_gran_to_debugging_csv[time_gran].drop([dropped_feature], axis=1)
                    except:
                        pass

            time_gran_to_debugging_csv[time_gran].to_csv(base_output_name + 'DEBUGGING_modz_feat_df_at_time_gran_of_'+\
                                                         str(time_gran) + '_sec.csv', na_rep='?')
            print "ide_angles", ide_train, ide_test

        list_of_feat_coefs_dfs.append(coef_feature_df)

    starts_of_testing_dict = {}
    for counter,name in enumerate(names):
        starts_of_testing_dict[name] = starts_of_testing[counter]

    starts_of_testing_df = pd.DataFrame(starts_of_testing_dict, index=['start_of_testing_phase'])

    print "list_of_rocs", list_of_rocs
    generate_report.generate_report(list_of_rocs, list_of_feat_coefs_dfs, list_of_attacks_found_dfs,
                                    recipes_used, base_output_name, time_grans, list_of_model_parameters,
                                    list_of_optimal_fone_scores, starts_of_testing_df, path_occurence_training_df,
                                    path_occurence_testing_df, percent_attacks, list_of_attacks_found_training_df,
                                    list_percent_attacks_training, feature_activation_heatmaps, feature_raw_heatmaps,
                                    ideal_thresholds, feature_activation_heatmaps_training, feature_raw_heatmaps_training,
                                    fraction_of_edge_weights, fraction_of_edge_pkts)

    print "multi_experiment_pipeline is all done! (NO ERROR DURING RUNNING)"
    #print "recall that this was the list of alert percentiles", percentile_thresholds
    return list_of_optimal_fone_scores,X_trains,Y_trains,X_tests,Y_tests, trained_models


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

