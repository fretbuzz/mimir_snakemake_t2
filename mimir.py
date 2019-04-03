### this file will eventually contain the CL interface for the system.
import argparse
import analysis_pipeline.config_parser

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This is the central CL interface for the MIMIR system')

    '''
    parser.add_argument('--training_pcap', dest='training_pcap', default=None,
                        help='this pcap is used as training data')
    parser.add_argument('--testing_pcap', dest='testing_pcap', default=None,
                        help='this pcap is used for testing data')

    parser.add_argument('--output_alert_file', dest='output_alert_file', default='./output_alert_file.csv',
                        help='alerts will be ouput to this file')
    parser.add_argument('--output_feature_csv', dest='output_feature_csv', default='./output_feature_csv.csv',
                        help='the feature dataframe is output here')

    # this needs to actually be integrated into the system at some point...
    parser.add_argument('--input_feature_csv', dest='input_feature_csv', default=None,
                        help='can input a feature csv for training purposes. this skips alot of processing')

    # these will probably be removed sometime relatively soon
    parser.add_argument('--trainTestRatio', dest='trainTestRatio', default='0.5',
                        help='will spit the training pcap into train/test data (temporally)')
    parser.add_argument('--no_testing_simattack', dest='no_testing_simattack_', action='store_true',
                       default=False,
                       help='do NOT simulate attacks in the testing data')
    '''

    parser.add_argument('--config_json', dest='config_json', default=None,
                        help='this is the configuration file used to process the file')
    args = parser.parse_args()
    if not args.config_json:
        analysis_pipeline.config_parser.parse_experimental_config('./analysis_pipeline/analysis_json/sockshop_one_v2.json')
    else:
        analysis_pipeline.config_parser.parse_experimental_config(args.config_json)


