from jinja2 import FileSystemLoader, Environment
import datetime
import pandas as pd
import pdfkit
import subprocess
import os


#####
## NOTE: this portion of the code is heavily influenced by:
## https://dev.to/goyder/automatic-reporting-in-python---part-2-from-hello-world-to-real-insights-8p3
###

def generate_report(list_of_rocs, list_of_feat_coef, list_of_attacks_found_dfs, recipes_used,
                    output_location, time_grans, list_of_model_parameters, list_of_optimal_fone_scores,
                    starts_of_testing_df, path_occurence_training_df, path_occurence_testing_df,
                    percent_attacks, list_of_attacks_found_training_df, percent_attacks_training,
                    feature_activation_heatmaps, feature_raw_heatmaps, ideal_thresholds,
                    feature_activation_heatmaps_training, feature_raw_heatmaps_training,
                    fraction_of_edge_weights, fraction_of_edge_pkts):
    # setup jinga and the associated template
    env = Environment(
        loader=FileSystemLoader(searchpath="src")
    )
    base_template = env.get_template("report_template.html")
    table_section_template = env.get_template("table_section.html")
    debug_section_template = env.get_template("debug_section.html")

    title = "MIMIR Results Report"
    #roc_placeholder = 'ROCS_GOES_HERE'
    #feature_table_placeholder = 'TABLE_OF_FEATURES_AND_COEFS_GOES_HERE'
    #attacks_found_placeholder = 'WHICH_RESULTS_FOUND_GO_HERE'
    sections = list()
    print "time_grans", time_grans
    for i in range(0, len(time_grans)):
        sections.append(table_section_template.render(
            time_gran= str(time_grans[i]) + " sec granularity",
            roc=list_of_rocs[i],
            feature_table=list_of_feat_coef[i].to_html(),
            attacks_found=list_of_attacks_found_dfs[i].to_html(),
            model_params=list_of_model_parameters[i],
            optimal_fOne = list_of_optimal_fone_scores[i],
            percent_attacks = percent_attacks[i],
            attacks_found_training = list_of_attacks_found_training_df[i].to_html(),
            percent_attacks_training = percent_attacks_training[i],
            feature_activation_heatmap = feature_activation_heatmaps[i],
            feature_raw_heatmap = feature_raw_heatmaps[i],
            ideal_threshold = ideal_thresholds[i],
            feature_activation_heatmap_training = feature_activation_heatmaps_training[i],
            feature_raw_heatmap_training = feature_raw_heatmaps_training[i]
        ))
    '''
    sections.append(table_section_template.render(
        time_gran= str(time_grans[1]) + " sec granularity",
        roc=list_of_rocs[1],
        feature_table=list_of_feat_coef[1].to_html(),
        attacks_found=list_of_attacks_found_dfs[1].to_html(),
        model_params=list_of_model_parameters[1],
    optimal_fOne = list_of_optimal_fone_scores[1]
    ))
    sections.append(table_section_template.render(
        time_gran= str(time_grans[2]) + " sec granularity",
        roc=list_of_rocs[2],
        feature_table=list_of_feat_coef[2].to_html(),
        attacks_found=list_of_attacks_found_dfs[2].to_html(),
        model_params=list_of_model_parameters[2],
    optimal_fOne = list_of_optimal_fone_scores[2]
    ))
    '''

    print "about to render the template..."
    #print sections
    #print base_template
    #print table_section_template

    date = str(datetime.datetime.now()) ### TODO: is this the right timezone?

    sections.append(debug_section_template.render(
        starts_of_testing_df = starts_of_testing_df.to_html(),
        path_occurence_training_df = path_occurence_training_df.transpose().to_html(),
        path_occurence_testing_df = path_occurence_testing_df.transpose().to_html()
    ))

    # render the template locally...
    with open("mulval_inouts/report.html", "w") as f:
        f.write(base_template.render(
            title=title,
            date = date,
            recipes_used = recipes_used,
            sections=sections,
            frac_exfil_weight = fraction_of_edge_weights,
            frac_exfil_pkts = fraction_of_edge_pkts
        ))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    #print "dir_path", dir_path
    config = pdfkit.configuration(wkhtmltopdf="/usr/local/bin/wkhtmltopdf")## TODO
    pdfkit.from_file("mulval_inouts/report.html", output_location + "_report.pdf", configuration=config)
    out = subprocess.check_output(['open', output_location + "_report.pdf"])
    print out

    ## note: if you wanna save an (e.g. archival) copy, that must be done manually. It is
    ## NOT done automatically.

if __name__ == "__main__":
    content = "Hello, world!"
    df_one = pd.DataFrame([1,2])
    df_two = pd.DataFrame([3,4])
    df_three = pd.DataFrame([5,6])
    list_of_rocs = ['roc1', 'roc2', 'roc3']
    recipes = ['wp1','wp2','wp4']
    recipes_used = " ,".join(str(x) for x in recipes)
    recipes_used = recipes_used[:-1]
    output_location = '.'
    time_grans = [1,2,3]
    list_of_model_parameters = [None, None, None]
    list_of_optimal_fone_scores = [0, 0, 0]
    generate_report(list_of_rocs, [df_one, df_two, df_three], [df_three, df_two, df_one],
                    recipes_used, output_location, time_grans, list_of_model_parameters,
                    list_of_optimal_fone_scores)