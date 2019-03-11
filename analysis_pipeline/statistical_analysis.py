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
                                         avg_exfil_per_min,
                                         avg_pkt_size,
                                         exfil_per_min_variance,
                                         pkt_size_variance):
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
    time_gran_to_attacks_found_df = {}
    time_gran_to_attacks_found_df_training = {}
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

    list_of_attacks_to_found_dfs = []
    list_of_attacks_to_found_training_df = []

    for time_gran,aggregate_mod_score_dfs in time_gran_to_aggregate_mod_score_dfs.iteritems():
        time_grans.append(time_gran)
        method_to_testing_df = {}
        method_to_training_df = {}

        X_train, y_train, X_test, y_test, pre_drop_X_train, time_gran_to_debugging_csv, dropped_feature_list, ide_train,\
            ide_test, X_train_exfil_weight, X_test_exfil_weight, exfil_paths, exfil_paths_train = \
            prepare_data(aggregate_mod_score_dfs, skip_model_part, ignore_physical_attacks_p, time_gran_to_debugging_csv, time_gran)

        # method_to_trainTest = {}
        # method_to_trainTest['ensemble_of_features'] = (X_train, X_test)
        # method_to_trainTest['ide_angle'] = (ide_train, ide_test)
        # method_to_trainTest['labels'] = (y_train, y_test)


        # TODO: probably want to pass in the feature selection capabilities as a first class function
        if 'lass_feat_sel' in base_output_name:
            X_train, X_test = lasso_feature_selection(X_train, y_train, X_test, y_test)

        X_trains[time_gran] = X_train
        Y_trains[time_gran] = y_train
        X_tests[time_gran] = X_test
        Y_tests[time_gran] = y_test

        dropped_columns = list(pre_drop_X_train.columns.difference(X_train.columns))

        ###########################
        # if I was going to change the model, this part would need to be what changes. For instance,
        # if I want to do logistic regresesion, this is probably where I'd need to do it (but would it
        # effect the information that I need to feed into the report generation component??) - in particular
        # the heatmap and ROC component
        clf.fit(X_train, y_train)

        #clf = sklearn.linear_model.LinearRegression()
        clf.fit(X_train, y_train)
        trained_models[time_gran] = clf
        score_val = clf.score(X_test, y_test)
        #print "score_val", score_val
        test_predictions = clf.predict(X=X_test)
        train_predictions = clf.predict(X=X_train)


        coef_dict = get_coef_dict(clf, X_train.columns.values, base_output_name, X_train.dtypes)
        print "coef_dict", coef_dict
        coef_feature_df = pd.DataFrame.from_dict(coef_dict, orient='index')
        coef_feature_df.columns = ['Coefficient']
        model_params = clf.get_params()
        try:
            model_params['alpha_val'] = clf.alpha_
        except:
            pass
        list_of_model_parameters.append(model_params)

        ####################################
        print '--------------------------'

        #print len(time_gran_to_debugging_csv[time_gran]["labels"]), len(np.concatenate([train_predictions, test_predictions]))
        #print len(time_gran_to_debugging_csv[time_gran].index)
        if not skip_model_part:
            time_gran_to_debugging_csv[time_gran].loc[:, "aggreg_anom_score"] = np.concatenate(
                [train_predictions, test_predictions])
        else:
            # time_gran_to_debugging_csv[time_gran].loc[:, "aggreg_anom_score"] = test_predictions
            time_gran_to_debugging_csv[time_gran].loc[:, "aggreg_anom_score"] = np.concatenate(
                [train_predictions, test_predictions])

        # make heatmaps so I can see which features are contributing
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

        current_heatmap_raw_val_path_training = base_output_name + 'training_raw_val_heatmap_' + str(time_gran) + '.png'
        local_heatmap_raw_val_path_training = 'temp_outputs/training_heatmap_raw_val_at_' +  str(time_gran) + '.png'
        current_heatmap_path_training = base_output_name + 'training_coef_act_heatmap_' + str(time_gran) + '.png'
        local_heatmap_path_training = 'temp_outputs/training_heatmap_coef_contribs_at_' +  str(time_gran) + '.png'
        coef_impact_df, raw_feature_val_df = generate_heatmap.generate_covariate_heatmap(coef_dict, X_train, exfil_paths_train)
        generate_heatmap.generate_heatmap(coef_impact_df, local_heatmap_path_training, current_heatmap_path_training)
        generate_heatmap.generate_heatmap(raw_feature_val_df, local_heatmap_raw_val_path_training, current_heatmap_raw_val_path_training)
        feature_activation_heatmaps_training.append('../' + local_heatmap_path_training)
        feature_raw_heatmaps_training.append('../' + local_heatmap_raw_val_path_training)

        ### step (3): Make ROCs and determine performance at optimal F1 operating point
        optimal_predictions, optimal_thresh, plot_path, ide_threshold = generate_ROC_curves(y_test, test_predictions, base_output_name,
                                                            time_gran, ide_test, ide_train, list_of_optimal_fone_scores)
        optimal_ide_prediction = [int(i > ide_threshold) for i in ide_test]
        ide_optimal_train_predictions = [int(i > ide_threshold) for i in ide_train]
        ideal_thresholds.append(optimal_thresh)
        list_of_rocs.append(plot_path)
        categorical_cm_df = determine_categorical_cm_df(y_test, optimal_predictions, exfil_paths, X_test_exfil_weight)
        list_of_attacks_found_dfs.append(categorical_cm_df)
        method_to_testing_df['mimir'] = categorical_cm_df

        optimal_train_predictions = [int(i > optimal_thresh) for i in train_predictions]
        categorical_cm_df_training = determine_categorical_cm_df(y_train, optimal_train_predictions, exfil_paths_train,
                                                                 X_train_exfil_weight)
        list_of_attacks_found_training_df.append(categorical_cm_df_training)
        method_to_training_df['mimir'] = categorical_cm_df_training


        ide_categorical_cm_df = determine_categorical_cm_df(y_test, optimal_ide_prediction, exfil_paths, X_test_exfil_weight)
        ide_categorical_cm_df_training = determine_categorical_cm_df(y_train, ide_optimal_train_predictions, exfil_paths_train,
                                                                 X_train_exfil_weight)
        method_to_training_df['ide'] = ide_categorical_cm_df_training
        method_to_testing_df['ide'] = ide_categorical_cm_df

        '''
        method_to_testing_df['mimir'] = categorical_cm_df
        method_to_training_df['mimir'] = categorical_cm_df_training
        '''

        if not skip_model_part:
            time_gran_to_debugging_csv[time_gran].loc[:, "anom_val_at_opt_pt"] = \
                np.concatenate([optimal_train_predictions, optimal_predictions])
        else:
            #time_gran_to_debugging_csv[time_gran].loc[:, "anom_val_at_opt_pt"] = optimal_predictions
            time_gran_to_debugging_csv[time_gran].loc[:, "anom_val_at_opt_pt"] = \
                np.concatenate([optimal_train_predictions, optimal_predictions])

        # I don't want the attributes w/ zero coefficients to show up in the debugging csv b/c it makes it hard to read
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

        list_of_feat_coefs_dfs.append(coef_feature_df)

        number_attacks_in_test = len(y_test[y_test['labels'] == 1])
        number_non_attacks_in_test = len(y_test[y_test['labels'] == 0])
        percent_attacks.append(float(number_attacks_in_test) / (number_non_attacks_in_test + number_attacks_in_test))
        number_attacks_in_train = len(y_train[y_train['labels'] == 1])
        number_non_attacks_in_train = len(y_train[y_train['labels'] == 0])
        list_percent_attacks_training.append(
            float(number_attacks_in_train) / (number_non_attacks_in_train + number_attacks_in_train))

        time_gran_to_attacks_found_df[time_gran] = method_to_testing_df
        time_gran_to_attacks_found_df_training[time_gran] = method_to_training_df


    starts_of_testing_dict = {}
    for counter,name in enumerate(names):
        starts_of_testing_dict[name] = starts_of_testing[counter]

    starts_of_testing_df = pd.DataFrame(starts_of_testing_dict, index=['start_of_testing_phase'])

    # TODO: this is kinda a bit of work.
    # per_attack_bar_graphs(method_to_results_df, temp_location, file_storage_location)
    #list_of_attacks_to_found_dfs, list_of_attacks_to_found_training_df

    print "list_of_rocs", list_of_rocs
    generate_report.generate_report(list_of_rocs, list_of_feat_coefs_dfs, list_of_attacks_found_dfs,
                                    recipes_used, base_output_name, time_grans, list_of_model_parameters,
                                    list_of_optimal_fone_scores, starts_of_testing_df, path_occurence_training_df,
                                    path_occurence_testing_df, percent_attacks, list_of_attacks_found_training_df,
                                    list_percent_attacks_training, feature_activation_heatmaps, feature_raw_heatmaps,
                                    ideal_thresholds, feature_activation_heatmaps_training, feature_raw_heatmaps_training,
                                    avg_exfil_per_min,avg_pkt_size, exfil_per_min_variance, pkt_size_variance)

    print "multi_experiment_pipeline is all done! (NO ERROR DURING RUNNING)"

    experiment_info = {}
    experiment_info["recipes_used"] = recipes_used
    experiment_info["avg_exfil_per_min"] = avg_exfil_per_min
    experiment_info["avg_pkt_size"] = avg_pkt_size
    experiment_info["exfil_per_min_variance"] = exfil_per_min_variance
    experiment_info["pkt_size_variance"] = pkt_size_variance

    return list_of_optimal_fone_scores,X_trains,Y_trains,X_tests,Y_tests, trained_models, time_gran_to_attacks_found_df, \
           time_gran_to_attacks_found_df_training,experiment_info

def drop_useless_columns_aggreg_DF(aggregate_mod_score_dfs):
    # '''

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
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='fraction_pod_comm_but_not_VIP_comms_')
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='fraction_pod_comm_but_not_VIP_comms_no_abs_')
    except:
        pass

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(
            columns='Communication Between Pods not through VIPs (no abs)_')  # might wanna just stop these from being generated...
    except:
        pass

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

    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Communication Between Pods not through VIPs (w abs)_')
    except:
        pass

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
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='ide_angles (w abs)_')
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
    ###############

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
    # Communication Not Through VIPs
    try:
        aggregate_mod_score_dfs = aggregate_mod_score_dfs.drop(columns='Communication Not Through VIPs')
    except:
        pass
        # '''
        # 'Communication Between Pods not through VIPs (no abs)_mod_z_score'
        # 'Fraction of Communication Between Pods not through VIPs (no abs)_mod_z_score'
    return aggregate_mod_score_dfs

# if we drop it in this function, it'll still show up in some of the debugging information that it wouldn't if
# we had dropped it in drop_useless_columns_aggreg_DF
'''
def drop_useless_columns_aggreg_testtrain_DF( aggregate_mod_score_dfs_training, aggregate_mod_score_dfs_testing):
    try:
        aggregate_mod_score_dfs_training = aggregate_mod_score_dfs_training.drop(columns='new_neighbors_outside')
        aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs_testing.drop(columns='new_neighbors_outside')
    except:
        pass
    try:
        aggregate_mod_score_dfs_training = aggregate_mod_score_dfs_training.drop(columns='new_neighbors_dns')
        aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs_testing.drop(columns='new_neighbors_dns')
    except:
        pass
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
    return aggregate_mod_score_dfs_training, aggregate_mod_score_dfs_testing
'''

def extract_comparison_methods(X_train, X_test):
    try:
        ewma_train = X_train['max_ewma_control_chart_scores']
        ewma_test = X_test['max_ewma_control_chart_scores']
        X_train = X_train.drop(columns='max_ewma_control_chart_scores')
        X_test = X_test.drop(columns='max_ewma_control_chart_scores')
    except:
        ewma_train = [0 for i in range(0, len(X_train))]
        ewma_test = [0 for i in range(0, len(X_test))]

    try:
        # if True:
        print X_train.columns
        ide_train = copy.deepcopy(X_train['ide_angles_'])
        #ide_train.fillna(ide_train.mean())
        print "ide_train", ide_train
        # exit(1222)
        copy_of_X_test = X_test.copy(deep=True)
        ide_test = copy.deepcopy(copy_of_X_test['ide_angles_'])

        #ide_test = ide_test.fillna(ide_train.mean())
        print "ide_test", ide_test
        X_train = X_train.drop(columns='ide_angles_')
        X_test = X_test.drop(columns='ide_angles_')
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

    return ide_train, ide_test, X_train, X_test

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

def get_coef_dict(clf, X_train_columns, base_output_name, X_train_dtypes):
    coef_dict = {}
    print "Coefficients: "
    print "LASSO model", clf.get_params()
    print '----------------------'
    print "len(clf.coef_)", len(clf.coef_), "len(X_train_columns)", len(X_train_columns)

    if 'logistic' in base_output_name:
        model_coefs = clf.coef_[0]
    else:
        model_coefs = clf.coef_

    if len(model_coefs) != (len(X_train_columns)):  # there is no plus one b/c the intercept is stored in clf.intercept_
        print "coef_ is different length than X_train_columns!", X_train_columns
        for counter, i in enumerate(X_train_dtypes):
            print counter, i, X_train_columns[counter]
            print model_coefs  # [counter]
            print len(model_coefs)
        exit(888)
    for coef, feat in zip(model_coefs, X_train_columns):
        coef_dict[feat] = coef
    print "COEFS_HERE"
    print "intercept...", float(clf.intercept_)
    coef_dict['intercept'] = float(clf.intercept_)
    for coef, feature in coef_dict.iteritems():
        print coef, feature

    return coef_dict

def prepare_data(aggregate_mod_score_dfs, skip_model_part, ignore_physical_attacks_p, time_gran_to_debugging_csv, time_gran):
    aggregate_mod_score_dfs = drop_useless_columns_aggreg_DF(aggregate_mod_score_dfs)

    if not skip_model_part:
        if ignore_physical_attacks_p:
            aggregate_mod_score_dfs = \
                aggregate_mod_score_dfs[~((aggregate_mod_score_dfs['labels'] == 1) &
                                          ((aggregate_mod_score_dfs['exfil_pkts'] == 0) &
                                           (aggregate_mod_score_dfs['exfil_weight'] == 0)))]

        aggregate_mod_score_dfs_training = aggregate_mod_score_dfs[aggregate_mod_score_dfs['is_test'] == 0]
        aggregate_mod_score_dfs_testing = aggregate_mod_score_dfs[aggregate_mod_score_dfs['is_test'] == 1]
        time_gran_to_debugging_csv[time_gran] = aggregate_mod_score_dfs.copy(deep=True)
        print "aggregate_mod_score_dfs_training", aggregate_mod_score_dfs_training
        print "aggregate_mod_score_dfs_testing", aggregate_mod_score_dfs_testing
        print aggregate_mod_score_dfs['is_test']

    else:
        ## note: generally you'd want to split into test and train sets, but if we're not doing logic
        ## part anyway, we just want quick-and-dirty results, so don't bother (note: so for formal purposes,
        ## DO NOT USE WITHOUT LOGIC CHECKING OR SOLVE THE TRAINING-TESTING split problem)
        aggregate_mod_score_dfs_training, aggregate_mod_score_dfs_testing = train_test_split(aggregate_mod_score_dfs,
                                                                                             test_size=0.5)
        time_gran_to_debugging_csv[time_gran] = aggregate_mod_score_dfs_training.copy(deep=True).append(
            aggregate_mod_score_dfs_testing.copy(deep=True))

    print aggregate_mod_score_dfs_training.index
    #aggregate_mod_score_dfs_training, aggregate_mod_score_dfs_testing = \
    #    drop_useless_columns_aggreg_testtrain_DF(aggregate_mod_score_dfs_training, aggregate_mod_score_dfs_testing)

    X_train = aggregate_mod_score_dfs_training.loc[:, aggregate_mod_score_dfs_training.columns != 'labels']
    y_train = aggregate_mod_score_dfs_training.loc[:, aggregate_mod_score_dfs_training.columns == 'labels']
    X_test = aggregate_mod_score_dfs_testing.loc[:, aggregate_mod_score_dfs_training.columns != 'labels']
    y_test = aggregate_mod_score_dfs_testing.loc[:, aggregate_mod_score_dfs_training.columns == 'labels']

    # get method to compare against and remove them from the DF...
    ide_train, ide_test, X_train, X_test = extract_comparison_methods(X_train, X_test)

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
           ide_train, ide_test, X_train_exfil_weight, X_test_exfil_weight, exfil_paths, exfil_paths_train


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

def generate_ROC_curves(y_test, test_predictions, base_output_name, time_gran, ide_test, ide_train, list_of_optimal_fone_scores):
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

    ax, _, plot_path = generate_alerts.construct_ROC_curve(list_of_x_vals, list_of_y_vals, title, ROC_path + plot_name, \
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

