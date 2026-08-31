#!/usr/bin/env nextflow
/*
 * Multi-omics biomarker signature for breast cancer relapse.
 *
 * Design note: the analysis scripts read and write a fixed project layout
 * (data/, results/) rather than passing matrices between steps as files. The
 * workflow therefore runs each stage against the mounted project directory and
 * threads a completion TOKEN between processes to enforce the dependency order.
 * The DAG below is the real execution order, and `-with-dag` renders it.
 */

nextflow.enable.dsl = 2

params.project  = "$projectDir/.."      // repo root
params.config   = "config/config.yaml"
params.subset   = 0                     // >0 => smoke-test cohort size
params.fast     = false                 // fewer folds / search candidates
params.skip_md5 = true

process VERIFY_DATA {
    tag "gate0"
    label 'light'
    input:  val ready
    output: val 'gate0', emit: token
    script:
    """
    cd ${params.project} && python scripts/00_verify_data.py \\
        --config ${params.config} ${params.skip_md5 ? '--skip-md5' : ''}
    """
}

process ASSEMBLE {
    tag "gate1"
    label 'heavy'
    input:  val token
    output: val 'gate1', emit: token
    script:
    """
    cd ${params.project} && python scripts/01_assemble.py --config ${params.config} ${params.subset > 0 ? "--subset-patients ${params.subset}" : ''}
    """
}

process HORIZON_SENSITIVITY {
    tag "gate1b"
    label 'light'
    input:  val token
    output: val 'gate1b', emit: token
    script:
    """
    cd ${params.project} && python scripts/01b_horizon_sensitivity.py --config ${params.config}
    """
}

process FETCH_GENESETS {
    tag "msigdb"
    label 'light'
    input:  val token
    output: val 'genesets', emit: token
    script:
    """
    cd ${params.project} && python scripts/fetch_genesets.py --config ${params.config}
    """
}

process MODEL_A {
    tag "gate2"
    label 'light'
    input:  val token
    output: val 'modelA', emit: token
    script:
    """
    cd ${params.project} && python scripts/02_model_a_clinical.py --config ${params.config}
    cd ${params.project} && python scripts/02b_baseline_diagnostics.py --config ${params.config}
    """
}

process MODEL_B {
    tag "gate3"
    label 'heavy'
    input:  val token
    output: val 'modelB', emit: token
    script:
    """
    cd ${params.project} && python scripts/03_model_b_xgboost.py --config ${params.config} ${params.fast ? '--fast' : ''}
    """
}

process MODEL_C {
    tag "gate4"
    label 'heavy'
    input:  val token_a
            val token_b
    output: val 'modelC', emit: token
    script:
    """
    cd ${params.project} && python scripts/04_model_c_pnet.py --config ${params.config} ${params.fast ? '--fast' : ''}
    """
}

process SIGNATURE {
    tag "gate5"
    label 'heavy'
    input:  val token
    output: val 'signature', emit: token
    script:
    """
    cd ${params.project} && python scripts/05_signature_reduction.py --config ${params.config} ${params.fast ? '--fast' : ''}
    """
}

process IMMUNE_DECONV {
    tag "gate6"
    label 'light'
    input:  val token
    output: val 'immune', emit: token
    script:
    """
    cd ${params.project} && python scripts/06_immune_deconv.py --config ${params.config}
    """
}

process EXTERNAL_METABRIC {
    tag "gate7"
    label 'heavy'
    input:  val token
    output: val 'metabric', emit: token
    script:
    """
    cd ${params.project} && python scripts/07_external_metabric.py --config ${params.config}
    """
}

process FIGURES {
    tag "figures"
    label 'light'
    input:  val tokens
    output: val 'figures', emit: token
    script:
    """
    cd ${params.project} && python scripts/08_figures.py --config ${params.config}
    """
}

process REPORT {
    tag "pdf"
    label 'light'
    input:  val token
    output: val 'report', emit: token
    script:
    """
    cd ${params.project} && python scripts/09_report.py --config ${params.config}
    """
}

workflow {
    start = Channel.value('go')

    VERIFY_DATA(start)
    ASSEMBLE(VERIFY_DATA.out.token)
    HORIZON_SENSITIVITY(ASSEMBLE.out.token)
    FETCH_GENESETS(VERIFY_DATA.out.token)

    MODEL_A(ASSEMBLE.out.token)
    MODEL_B(MODEL_A.out.token)
    MODEL_C(ASSEMBLE.out.token, FETCH_GENESETS.out.token)

    SIGNATURE(MODEL_B.out.token.mix(MODEL_C.out.token).collect())
    IMMUNE_DECONV(SIGNATURE.out.token)
    EXTERNAL_METABRIC(SIGNATURE.out.token)

    FIGURES(IMMUNE_DECONV.out.token.mix(EXTERNAL_METABRIC.out.token,
                                        HORIZON_SENSITIVITY.out.token).collect())
    REPORT(FIGURES.out.token)
}
