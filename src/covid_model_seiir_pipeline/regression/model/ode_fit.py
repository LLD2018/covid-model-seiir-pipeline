# -*- coding: utf-8 -*-
"""
    ODE Process
    ~~~~~~~~~~~~
"""
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
from odeopt.ode import RK4
from odeopt.ode import LinearFirstOrder
from odeopt.core.utils import linear_interpolate


class SingleGroupODEProcess:

    def __init__(self, df,
                 col_date,
                 col_cases,
                 col_pop,
                 col_loc_id,
                 today,
                 day_shift,
                 lag_days,
                 alpha,
                 sigma,
                 gamma1,
                 gamma2,
                 solver_class=RK4,
                 solver_dt=1e-1):
        """Constructor of the SingleGroupODEProcess

        Args:
            df (pd.DateFrame): DataFrame contains all the data.
            col_date (str): Date of the rows.
            col_cases (str): Column with new infectious data.
            col_pop (str): Column with population.
            col_loc_id (str): Column with location id.
            today (np.datetime64): Indicating when "today" is. Defaults to the actual today.
            day_shift (arraylike): Days shift for the data sub-selection.
            alpha (arraylike): bounds for uniformly sampling alpha.
            sigma (arraylike): bounds for uniformly sampling sigma.
            gamma1 (arraylike): bounds for uniformly sampling gamma1.
            gamma2 (arraylike): bounds for uniformly sampling gamma2.
            solver_class (ODESolver, optional): Solver for the ODE system.
            solver_dt (float, optional): Step size for the ODE system.
        """
        # observations
        assert col_date in df
        assert col_cases in df
        assert col_pop in df
        assert col_loc_id in df
        self.col_date = col_date
        self.col_cases = col_cases
        self.col_pop = col_pop
        self.col_loc_id = col_loc_id

        self.loc_id = df[self.col_loc_id].values[0]

        self.alpha = alpha
        self.sigma = sigma
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.N = df[self.col_pop].values[0]

        self.today = today
        self.day_shift = day_shift
        self.lag_days = lag_days

        # subset the data
        df.sort_values(self.col_date, inplace=True)
        date = pd.to_datetime(df[col_date])
        # cast this from numpy.int64 to int because travis/tox was erroring on
        # this, only for python 3.7, with a ValueError
        end_date = self.today + np.timedelta64(int(self.day_shift - self.lag_days), 'D')
        # Sometimes we don't have leading indicator data, so the day shift
        # will put us into padded zeros.  Correct for this.
        max_end_date = date[df[col_cases] > 0].max()
        end_date = min(end_date, max_end_date)

        idx = date <= end_date

        cases_threshold = 50.0
        start_date = date[df[col_cases] >= cases_threshold].min()
        idx_final = idx & (date >= start_date)
        infection_end_date = self.today - pd.Timedelta(days=self.lag_days)
        while np.sum(idx_final) <= 2 or infection_end_date < start_date:
            cases_threshold *= 0.5
            print(f'reduce cases threshold for {self.loc_id} to'
                  f'{cases_threshold}')
            start_date = date[df[col_cases] >= cases_threshold].min()
            idx_final = idx & (date >= start_date)
            if cases_threshold < 1e-6:
                # this is a data poor location, so we just use the whole time
                # series.
                start_date = date.min()
                idx_final = idx
                break

        self.df = df[idx_final].copy()
        date = date[idx_final]

        # save start and end date
        self.start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        self.end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        # compute days
        self.col_days = 'days'
        self.df[self.col_days] = (date - date.min()).dt.days.values

        # parse input
        self.date = self.df[self.col_date]
        self.t = self.df[self.col_days].values
        self.obs = self.df[self.col_cases].values
        self.init_cond = {
            'S': self.N,
            'E': self.obs[0],
            'I1': 1.0,
            'I2': 0.0,
            'R': 0.0
        }

        # ode solver setup
        self.solver_class = solver_class
        self.solver_dt = solver_dt
        self.t_span = np.array([np.min(self.t), np.max(self.t)])
        self.t_params = np.arange(self.t_span[0],
                                  self.t_span[1] + self.solver_dt,
                                  self.solver_dt)
        self.params = None
        self.components = None
        self.create_ode_sys()

    def create_ode_sys(self):
        """Create 1D ODE solver.
        """
        self.step_ode_sys = LinearFirstOrder(
            self.sigma,
            solver_class=self.solver_class,
            solver_dt=self.solver_dt
        )

    def process(self):
        """Process the data.
        """
        self.rhs_newE = linear_interpolate(self.t_params, self.t, self.obs)
        # fit the E
        self.step_ode_sys.update_given_params(c=self.sigma)
        E = self.step_ode_sys.simulate(self.t_params,
                                       np.array([self.init_cond['E']]),
                                       self.t_params,
                                       self.rhs_newE[None, :])[0]

        # fit I1
        self.step_ode_sys.update_given_params(c=self.gamma1)
        # modify initial condition of I1
        self.init_cond.update({
            'I1': (self.rhs_newE[0]/5.0)**(1.0/self.alpha)
        })
        I1 = self.step_ode_sys.simulate(self.t_params,
                                        np.array([self.init_cond['I1']]),
                                        self.t_params,
                                        self.sigma*E[None, :])[0]

        # fit I2
        self.step_ode_sys.update_given_params(c=self.gamma2)
        I2 = self.step_ode_sys.simulate(self.t_params,
                                        np.array([self.init_cond['I2']]),
                                        self.t_params,
                                        self.gamma1*I1[None, :])[0]

        # fit S
        self.init_cond.update({
            'S': self.N - self.init_cond['E'] - self.init_cond['I1']
        })
        self.step_ode_sys.update_given_params(c=0.0)
        S = self.step_ode_sys.simulate(self.t_params,
                                       np.array([self.init_cond['S']]),
                                       self.t_params,
                                       -self.rhs_newE[None, :])[0]
        neg_S_idx = S < 0.0

        # fit R
        self.step_ode_sys.update_given_params(c=0.0)
        R = self.step_ode_sys.simulate(self.t_params,
                                       np.array([self.init_cond['R']]),
                                       self.t_params,
                                       self.gamma2*I2[None, :])[0]

        if np.any(neg_S_idx):
            id_min = np.min(np.arange(S.size)[neg_S_idx])
            S[id_min:] = S[id_min - 1]
            E[id_min:] = E[id_min - 1]
            I1[id_min:] = I1[id_min - 1]
            I2[id_min:] = I2[id_min - 1]
            R[id_min:] = R[id_min - 1]

        self.components = {
            'S': S,
            'E': E,
            'I1': I1,
            'I2': I2,
            'R': R
        }

        # get beta
        self.params = (self.rhs_newE/
                       ((S/self.N)*
                        ((I1 + I2)**self.alpha)))[None, :]

    def predict(self, t):
        params = linear_interpolate(t, self.t_params, self.params)
        components = {
            c: linear_interpolate(t, self.t_params, self.components[c])
            for c in self.components
        }
        components.update({
            'newE': linear_interpolate(t, self.t_params, self.rhs_newE)
        })
        components.update({
            'newE_obs': linear_interpolate(t, self.t, self.obs)
        })
        return params, components

    def create_result_df(self):
        """Create result DataFrame.
        """
        params, components = self.predict(self.t)
        df_result = pd.DataFrame({
            self.col_loc_id: self.loc_id,
            self.col_date: self.date,
            'days': self.t,
            'beta': params[0]
        })

        for k, v in components.items():
            df_result[k] = v

        return df_result.rename(columns={self.col_loc_id: "location_id"})

    def create_params_df(self):
        """Create parameter DataFrame.
        """
        df_params = pd.DataFrame([self.alpha, self.sigma,
                                  self.gamma1, self.gamma2],
                                 index=['alpha', 'sigma', 'gamma1', 'gamma2'],
                                 columns=['params'])

        return df_params

    def create_start_end_date_df(self):
        df_result = pd.DataFrame({
            self.col_loc_id: self.loc_id,
            'start_date': self.start_date,
            'end_date': self.end_date
        }, index=[0])
        return df_result


@dataclass
class ODEProcessInput:
    df_dict: Dict
    col_date: str
    col_cases: str
    col_pop: str
    col_loc_id: str
    col_lag_days: str
    col_observed: str

    alpha: pd.Series
    sigma: pd.Series
    gamma1: pd.Series
    gamma2: pd.Series
    solver_dt: float
    day_shift: pd.Series


class ODEProcess:
    """ODE Process for multiple group.
    """
    def __init__(self, ode_process_input: ODEProcessInput):
        """Constructor of ODEProcess.

        Args:
            ode_process_input: ODEProcessInput
        """
        self.df_dict = ode_process_input.df_dict
        self.col_date = ode_process_input.col_date
        self.col_cases = ode_process_input.col_cases
        self.col_pop = ode_process_input.col_pop
        self.col_loc_id = ode_process_input.col_loc_id
        self.col_lag_days = ode_process_input.col_lag_days
        self.col_observed = ode_process_input.col_observed

        self.solver_dt = ode_process_input.solver_dt

        # create the location id
        self.loc_ids = np.sort(list(self.df_dict.keys()))

        self.alpha = ode_process_input.alpha
        self.sigma = ode_process_input.sigma
        self.gamma1 = ode_process_input.gamma1
        self.gamma2 = ode_process_input.gamma2
        self.day_shift = ode_process_input.day_shift.astype(int)

        # lag days
        self.lag_days = self.df_dict[self.loc_ids[0]][
            self.col_lag_days].values[0]
        self.today_dict = {
            loc_id: np.datetime64(
                self.df_dict[loc_id].loc[self.df_dict[loc_id][self.col_observed] == 1, self.col_date].max()
            )
            for loc_id in self.loc_ids
        }

        # create model for each location
        self.models = {}
        for loc_id in self.loc_ids:
            self.models[loc_id] = SingleGroupODEProcess(
                self.df_dict[loc_id],
                self.col_date,
                self.col_cases,
                self.col_pop,
                self.col_loc_id,
                day_shift=self.day_shift.loc[loc_id],
                lag_days=self.lag_days,
                alpha=self.alpha.loc[loc_id],
                sigma=self.sigma.loc[loc_id],
                gamma1=self.gamma1.loc[loc_id],
                gamma2=self.gamma2.loc[loc_id],
                solver_dt=self.solver_dt,
                today=self.today_dict[loc_id],
            )

    def process(self):
        """Process all models.
        """
        for loc_id, model in self.models.items():
            model.process()
        return pd.concat([
            model.create_result_df()
            for loc_id, model in self.models.items()
        ])

    def create_params_df(self):
        """Create parameter DataFrame.
        """
        df_params = pd.concat(
            [self.alpha, self.sigma, self.gamma1, self.gamma2, self.day_shift],
            axis=1
        ).reset_index()
        return df_params

    def create_start_end_date_df(self):
        """
        Create starting and ending date data frames for data used to fit, by group.
        """
        return pd.concat([
            model.create_start_end_date_df()
            for loc_id, model in self.models.items()
        ]).reset_index(drop=True)
