"""Runner for the beta ODE fit."""
from pathlib import Path

from covid_shared import cli_tools
from loguru import logger

from covid_model_seiir_pipeline.ode_fit import FitSpecification
from covid_model_seiir_pipeline import paths
from covid_model_seiir_pipeline.ode_fit.data import ODEDataInterface
from covid_model_seiir_pipeline.ode_fit.workflow import ODEFitWorkflow


def do_beta_fit(app_metadata: cli_tools.Metadata,
                fit_specification: FitSpecification):
    logger.debug('Starting Beta fit.')

    # init high level objects
    ode_paths = paths.ODEPaths(Path(fit_specification.data.output_root), read_only=False)
    infection_paths = paths.InfectionPaths(Path(fit_specification.data.infection_version))

    data_interface = ODEDataInterface(
        ode_paths=ode_paths,
        infection_paths=infection_paths,
    )

    # Grab canonical location list from arguments
    location_metadata = data_interface.load_location_ids_from_primary_source(
        location_set_version_id=fit_specification.data.location_set_version_id,
        location_file=fit_specification.data.location_set_file
    )
    # Filter to the intersection of what's available from the infection data.
    location_ids = data_interface.filter_location_ids(location_metadata)
    # Setup directory structure and save location and specification data.
    ode_paths.make_dirs(location_ids)
    data_interface.dump_location_ids(location_ids)
    # Fixme: Inconsistent data writing interfaces
    fit_specification.dump(ode_paths.fit_specification)

    # build workflow and launch
    ode_wf = ODEFitWorkflow(fit_specification.data.output_root)
    ode_wf.attach_ode_tasks(fit_specification.parameters.n_draws)
    ode_wf.run()