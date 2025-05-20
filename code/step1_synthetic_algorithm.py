import sdv
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from sdv.datasets.local import load_csvs
from sdv.datasets.demo import download_demo
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer 
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot


def step1_synthetic_algorithm(filepath):
    print("Running step1_synthetic_algorithm")

    os.makedirs(os.path.join(filepath, 'data'), exist_ok=True)
    os.makedirs(os.path.join(filepath, 'results/SDV'), exist_ok=True)
        
    df_train = pd.read_csv(os.path.join(filepath, 'GS_gwas/gene_scores_train_gwas.csv')) 
    df_val = pd.read_csv(os.path.join(filepath, 'GS_gwas/gene_scores_val_gwas.csv')) 
    df = pd.concat([df_train,df_val]).reset_index(drop=True)
    df.to_csv(os.path.join(filepath,'data/synthetic.csv'), index=False)

    # Load data to SDV
    FOLDER_NAME = os.path.join(filepath,'data')
    data = load_csvs(folder_name=FOLDER_NAME)
    real_data = data['synthetic']

    # Write SDV metadata
    selected_genes = [col for col in data['synthetic'].columns if col not in ['IID', 'outcome']]

    # Create JSON mannually
    json_data = {
        "tables": {
            "selected_data": {
                "columns": {
                    "IID": {"sdtype": "id"},
                    "outcome": {"sdtype": "categorical"},
                },
                "primary_key": "IID"
            }
        },
        "relationships": [],
        "METADATA_SPEC_VERSION": "V1"
    }

    # Add gene columns with numerical sdtype
    for gene in selected_genes:
        json_data["tables"]["selected_data"]["columns"][gene] = {"sdtype": "numerical"}

    # Save JSON file
    with open(os.path.join(FOLDER_NAME,"metadata.json"), "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    # Load metadata
    metadata = Metadata.load_from_json(os.path.join(FOLDER_NAME,'metadata.json'))

    # Gaussian copula synthesizer
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(num_rows=100000) 
    synthetic_data.to_csv(os.path.join(filepath,'data/synth_GC.csv'), index=False)

    quality_report = evaluate_quality(
        real_data,
        synthetic_data,
        metadata)
    quality_report.save(os.path.join(filepath, 'results/SDV/GC_quality_report.json'))

    fig = get_column_plot(
        real_data,
        synthetic_data,
        column_name='outcome',
        metadata=metadata
    )
    fig.write_html(os.path.join(filepath, 'results/SDV/GC_column_plot.html'))
    #fig.savefig(os.path.join(filepath, 'results/SDV/GC_column_plot.png'), dpi=300, bbox_inches='tight')
    #plt.close(fig)

    # CTGAN Synthesizer
    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(num_rows=100000)
    synthetic_data.to_csv(os.path.join(filepath,'data/synth_CTGAN.csv'), index=False)

    quality_report = evaluate_quality(
        real_data,
        synthetic_data,
        metadata)
    quality_report.save(os.path.join(filepath, 'results/SDV/CTGAN_quality_report.json'))

    fig = get_column_plot(
        real_data,
        synthetic_data,
        column_name='outcome',
        metadata=metadata
    )
    fig.write_html(os.path.join(filepath, 'results/SDV/CTGAN_column_plot.html'))
    #fig.savefig(os.path.join(filepath, 'results/SDV/CTGAN_column_plot.png'), dpi=300, bbox_inches='tight')
    #plt.close(fig)

    #train a model and create synthetic data.
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(num_rows=100000)
    synthetic_data.to_csv(os.path.join(filepath,'data/synth_TVAE.csv'), index=False)

    quality_report = evaluate_quality(
        real_data,
        synthetic_data,
        metadata)
    quality_report.save(os.path.join(filepath, 'results/SDV/TVAE_quality_report.json'))

    fig = get_column_plot(
        real_data,
        synthetic_data,
        column_name='outcome',
        metadata=metadata
    )
    fig.write_html(os.path.join(filepath, 'results/SDV/TVAE_column_plot.html'))
    #fig.savefig(os.path.join(filepath, 'results/SDV/TVAE_column_plot.png'), dpi=300, bbox_inches='tight')
    #plt.close(fig)

    print("step1_synthetic_algorithm completed")
