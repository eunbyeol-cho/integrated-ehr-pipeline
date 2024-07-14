import sys
import os
import re
import shutil
import subprocess
import logging
import pickle
import h5py

import pandas as pd
import numpy as np
import pyspark.sql.functions as F

from typing import Union, List
from functools import reduce
from itertools import chain
from transformers import AutoTokenizer
from tqdm import tqdm
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType


logger = logging.getLogger(__name__)

class EHR(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.cache = cfg.cache
        cache_dir = os.path.expanduser("~/.cache/ehr")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

        if self.cache:
            logger.warn(
                "--cache is set to True. Note that it forces to load cached"
                " data from {},".format(cache_dir)
                + " which may ignore some arguments such as --first_icu, as well as task related arguments (--mortality, --los_3day, etc.)"
                " If you want to avoid this, do not set --cache to True."
            )

        self.data_dir = cfg.data
        self.ccs_path = cfg.ccs
        self.gem_path = cfg.gem
        self.ext = cfg.ext

        self.max_event_size = (
            cfg.max_event_size if cfg.max_event_size is not None else sys.maxsize
        )
        self.min_event_size = (
            cfg.min_event_size if cfg.min_event_size is not None else 1
        )
        assert self.min_event_size > 0, (
            "--min_event_size could not be negative or zero", self.min_event_size
        )
        assert self.min_event_size <= self.max_event_size, (
            self.min_event_size,
            self.max_event_size,
        )

        self.max_event_token_len = cfg.max_event_token_len
        self.max_patient_token_len = cfg.max_patient_token_len

        self.max_age = cfg.max_age if cfg.max_age is not None else sys.maxsize
        self.min_age = cfg.min_age if cfg.min_age is not None else 0
        assert self.min_age <= self.max_age, (self.min_age, self.max_age)

        self.obs_size = cfg.obs_size
        self.gap_size = cfg.gap_size
        self.pred_size = cfg.pred_size
        self.long_term_pred_size = cfg.long_term_pred_size

        self.first_icu = cfg.first_icu

        # tasks
        self.mortality = cfg.mortality
        self.long_term_mortality = cfg.long_term_mortality
        self.los_3day = cfg.los_3day
        self.los_7day = cfg.los_7day
        self.readmission = cfg.readmission
        self.final_acuity = cfg.final_acuity
        self.imminent_discharge = cfg.imminent_discharge
        self.diagnosis = cfg.diagnosis
        self.creatinine = cfg.creatinine
        self.bilirubin = cfg.bilirubin
        self.platelets = cfg.platelets
        self.wbc = cfg.wbc
        self.hb = cfg.hb
        self.bicarbonate = cfg.bicarbonate
        self.sodium = cfg.sodium

        self.chunk_size = cfg.chunk_size

        self.dest = cfg.dest
        self.valid_percent = cfg.valid_percent
        self.seed = [int(s) for s in cfg.seed.replace(' ','').split(",")]
        assert 0 <= cfg.valid_percent and cfg.valid_percent <= 0.5

        self.bins = cfg.bins

        self.special_tokens_dict = dict()
        self.max_special_tokens = 100

        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        self.table_type_id = 1
        self.column_type_id = 2
        self.value_type_id = 3
        self.timeint_type_id = 4
        self.cls_type_id = 5
        self.sep_type_id = 6

        self.others_dpe_id = 14

        self._icustay_fname = None
        self._patient_fname = None
        self._admission_fname  = None
        self._diagnosis_fname = None

        self._icustay_key = None
        self._hadm_key = None

        self.max_time_len = 3 if self.obs_size > 12 else 2

    @property
    def icustay_fname(self):
        return self._icustay_fname
    
    @property
    def patient_fname(self):
        return self._patient_fname
    
    @property
    def admission_fname(self):
        return self._admission_fname
    
    @property
    def diagnosis_fname(self):
        return self._diagnosis_fname

    @property
    def icustay_key(self):
        return self._icustay_key
    
    @property
    def hadm_key(self):
        return self._hadm_key
    
    @property
    def patient_key(self):
        return self._patient_key

    @property
    def determine_first_icu(self):
        return self._determine_first_icu

    @property
    def num_special_tokens(self):
        return len(self.special_tokens_dict)

    def build_cohorts(self, icustays, cached=False):
        if cached:
            cohorts = self.load_from_cache(self.ehr_name + ".cohorts")
            if cohorts is not None:
                return cohorts

        if not self.is_compatible(icustays):
            raise AssertionError(
                "{} do not have required columns to build cohorts.".format(self.icustay_fname)
                + " Please make sure that dataframe for icustays is compatible with other ehrs."
            )

        logger.info(
            "Start building cohorts for {}".format(self.ehr_name)
        )

        obs_size = self.obs_size
        gap_size = self.gap_size

        icustays = icustays[icustays["LOS"] >= (obs_size + gap_size) / 24]
        icustays = icustays[
            (self.min_age <= icustays["AGE"]) & (icustays["AGE"] <= self.max_age)
        ]
        icustays['GENDER'] = icustays['gender'].astype(str)

        # we define labels for the readmission task in this step
        # since it requires to observe each next icustays,
        # which would have been excluded in the final cohorts
        icustays.sort_values([self.hadm_key, self.icustay_key], inplace=True)
        
        if self.readmission:
            icustays["readmission"] = 1
            icustays.loc[
                icustays.groupby(self.hadm_key)[self.determine_first_icu].idxmax(),
                "readmission"
            ] = 0
        if self.first_icu:
            icustays = icustays.loc[
                icustays.groupby(self.hadm_key)[self.determine_first_icu].idxmin()
            ]

        logger.info(
            "cohorts have been built successfully. Loaded {} cohorts.".format(
                len(icustays)
            )
        )
        self.save_to_cache(icustays, self.ehr_name + ".cohorts")

        return icustays

    # TODO process specific tasks according to user choice?
    def prepare_tasks(self, cohorts, spark, cached=False):
        if cached:
            labeled_cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled")
            if labeled_cohorts is not None:
                return labeled_cohorts
            else:
                raise RuntimeError()

        logger.info(
            "Start labeling cohorts for predictive tasks."
        )
        
        labeled_cohorts = cohorts[[
            self.hadm_key,
            self.icustay_key,
            self.patient_key,
            "readmission",
            "LOS",
            "INTIME",
            "OUTTIME",
            "DISCHTIME",
            "IN_ICU_MORTALITY",
            "HOS_DISCHARGE_LOCATION",
            "GENDER",
            "AGE",
        ]].copy()

        # mortality prediction
        # if the discharge location of an icustay is 'Death'
        #   & intime + obs_size + gap_size <= dischtime <= intime + obs_size + pred_size
        # it is assigned positive label on the mortality prediction

        if self.mortality:
            labeled_cohorts["mortality"] = (
                (
                    (labeled_cohorts["IN_ICU_MORTALITY"] == "Death")
                    | (labeled_cohorts["HOS_DISCHARGE_LOCATION"] == "Death")
                )
                & (
                    self.obs_size * 60 + self.gap_size * 60 < labeled_cohorts["DISCHTIME"]
                )
                & (
                    labeled_cohorts["DISCHTIME"] <= self.obs_size * 60 + self.pred_size * 60
                )
            ).astype(int)

        if self.long_term_mortality:
            labeled_cohorts["long_term_mortality"] = (
                (
                    (labeled_cohorts["IN_ICU_MORTALITY"] == "Death")
                    | (labeled_cohorts["HOS_DISCHARGE_LOCATION"] == "Death")
                )
                & (
                    self.obs_size * 60 + self.gap_size * 60 < labeled_cohorts["DISCHTIME"]
                )
                & (
                    labeled_cohorts["DISCHTIME"] <= self.obs_size * 60 + self.long_term_pred_size * 60
                )
            ).astype(int)

        if self.los_3day:
            labeled_cohorts["los_3day"] = (labeled_cohorts["LOS"] > 3).astype(int)
        if self.los_7day:
            labeled_cohorts["los_7day"] = (labeled_cohorts["LOS"] > 7).astype(int)

        if self.final_acuity or self.imminent_discharge:

            # if the discharge of 'Death' occurs in icu or hospital
            # we retain these cases for the imminent discharge task
            labeled_cohorts["IN_HOSPITAL_MORTALITY"] = (
                (~labeled_cohorts["IN_ICU_MORTALITY"])
                & (labeled_cohorts["HOS_DISCHARGE_LOCATION"] == "Death")
            ).astype(int)

            if self.final_acuity:

                # define final acuity prediction task
                labeled_cohorts["final_acuity"] = labeled_cohorts["HOS_DISCHARGE_LOCATION"]
                labeled_cohorts.loc[
                    labeled_cohorts["IN_ICU_MORTALITY"] == 1, "final_acuity"
                ] = "IN_ICU_MORTALITY"
                labeled_cohorts.loc[
                    labeled_cohorts["IN_HOSPITAL_MORTALITY"] == 1, "final_acuity"
                ] = "IN_HOSPITAL_MORTALITY"
                # NOTE we drop null value samples #TODO

                with open(os.path.join(self.dest, self.ehr_name + "_final_acuity_classes.tsv"), "w") as f:
                    for i, cat in enumerate(
                        labeled_cohorts["final_acuity"].astype("category").cat.categories
                    ):
                        print("{}\t{}".format(i, cat), file=f)
                labeled_cohorts["final_acuity"] = (
                    labeled_cohorts["final_acuity"].astype("category").cat.codes
                )

            if self.imminent_discharge:
            # define imminent discharge prediction task
                is_discharged = (
                    (
                        self.obs_size * 60 + self.gap_size * 60 <= labeled_cohorts["DISCHTIME"]
                    )
                    & (
                        labeled_cohorts["DISCHTIME"] <= self.obs_size * 60 + self.pred_size * 60)
                )
                labeled_cohorts.loc[is_discharged, "imminent_discharge"] = labeled_cohorts.loc[
                    is_discharged, "HOS_DISCHARGE_LOCATION"
                ]
                labeled_cohorts.loc[
                    is_discharged & (
                        (labeled_cohorts["IN_ICU_MORTALITY"] == 1)
                        | (labeled_cohorts["IN_HOSPITAL_MORTALITY"] == 1)
                    ),
                    "imminent_discharge"
                ] = "Death"
                labeled_cohorts.loc[~is_discharged, "imminent_discharge"] = "No Discharge"
                # NOTE we drop null value samples #TODO

                with open(
                    os.path.join(self.dest, self.ehr_name + "_imminent_discharge_classes.tsv"), "w"
                ) as f:
                    for i, cat in enumerate(
                        labeled_cohorts["imminent_discharge"].astype("category").cat.categories
                    ):
                        print("{}\t{}".format(i, cat), file=f)
                labeled_cohorts["imminent_discharge"] = (
                    labeled_cohorts["imminent_discharge"].astype("category").cat.codes
                )

            labeled_cohorts = labeled_cohorts.drop(
                columns=["IN_HOSPITAL_MORTALITY"]
            )

        # clean up unnecessary columns
        labeled_cohorts = labeled_cohorts.drop(
            columns=[
                "LOS",
                "IN_ICU_MORTALITY",
                "DISCHTIME",
                "HOS_DISCHARGE_LOCATION"
            ]
        )

        self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")

        logger.info("Done preparing tasks except for diagnosis prediction.")

        return labeled_cohorts


    def process_tables(self, cohorts, spark):
        # in: cohorts, sparksession
        # out: Spark DataFrame with (stay_id, time offset, inp, type, dpe)
        if isinstance(cohorts, pd.DataFrame):
            # cohorts = cohorts[[self.hadm_key, self.icustay_key, "INTIME", "OUTTIME"]]
            cohorts = cohorts[[self.hadm_key, self.icustay_key, "INTIME", "OUTTIME", "AGE", "GENDER", "long_term_mortality"]]
            logger.info("Start Preprocessing Tables, Cohort Numbers: {}".format(len(cohorts)))
            cohorts = spark.createDataFrame(cohorts)
            print("Converted Cohort to Pyspark DataFrame")
        else:
            logger.info("Start Preprocessing Tables")
            

        events_dfs = []
        vocab_dict = dict() 
        for table in self.tables:
            fname = table["fname"]
            table_name = fname.split('/')[-1][: -len(self.ext)]
            timestamp_key = table["timestamp"]
            excludes = table["exclude"]
            obs_size = self.obs_size
            gap_size = self.gap_size
            logger.info("{} in progress.".format(fname))

            code_to_descriptions = None
            if "code" in table:
                code_to_descriptions = {
                    k: pd.read_csv(os.path.join(self.data_dir, v))
                    for k, v in zip(table["code"], table["desc"])
                }
                code_to_descriptions = {
                    k: dict(zip(v[k], v[d_k]))
                    for (k, v), d_k in zip(
                        code_to_descriptions.items(), table["desc_key"]
                    )
                }

            infer_icustay_from_hadm_key = False

            events = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
            if self.icustay_key not in events.columns:
                infer_icustay_from_hadm_key = True
                if self.hadm_key not in events.columns:
                    raise AssertionError(
                        "{} doesn't have one of these columns: {}".format(
                            fname, [self.icustay_key, self.hadm_key]
                        )
                    )

            events = events.drop(*excludes)
            if table["timeoffsetunit"]=='abs':
                events = events.withColumn(timestamp_key, F.to_timestamp(timestamp_key))

            if infer_icustay_from_hadm_key:
                events = events.join(
                        cohorts.select(self.hadm_key, self.icustay_key, "INTIME", "OUTTIME"),
                        on=self.hadm_key, how="inner"
                    )
                if table["timeoffsetunit"] =='abs':
                    events = (
                        events.withColumn(
                            "TEMP_TIME",
                            F.round((F.col(timestamp_key).cast("long") - F.col("INTIME").cast("long")) / 60)
                        ).filter(F.col("TEMP_TIME") >= 0)
                        .filter(F.col("TEMP_TIME")<=F.col("OUTTIME"))
                        .drop("TEMP_TIME")
                    )
                else:
                    # All tables in eICU has icustay_key -> no need to handle
                    raise NotImplementedError()
                events = events.join(cohorts.select(self.icustay_key), on=self.icustay_key, how='leftsemi')

            else:
                events = events.join(cohorts.select(self.icustay_key, "INTIME", "OUTTIME"), on=self.icustay_key, how="inner")

            if table["timeoffsetunit"] == 'abs':
                events = events.withColumn("TIME", F.round((F.col(timestamp_key).cast("long") - F.col("INTIME").cast("long")) / 60))
                events = events.drop(timestamp_key)
            elif table["timeoffsetunit"] == "min":
                events = events.withColumn("TIME", F.col(timestamp_key).cast("int"))
            else:
                raise NotImplementedError()

            events = events.filter(F.col("TIME") >= 0).filter(F.col("TIME") <= obs_size * 60)

            events = events.drop("INTIME", "OUTTIME", self.hadm_key)

            if code_to_descriptions:
                for col in code_to_descriptions.keys():
                    mapping_expr = F.create_map([F.lit(x) for x in chain(*code_to_descriptions[col].items())])
                    events = events.withColumn(col, mapping_expr[F.col(col)])

            def process_unit(text, type_id):
                # Given (table_name|col|val), generate ([inp], [type], [dpe])
                text = re.sub(r"\d*\.\d+", lambda x: str(round(float(x.group(0)), 4)), str(text))
                number_groups = [g for g in re.finditer(r"([0-9]+([.][0-9]*)?|[0-9]+|\.+)", text)]
                text = re.sub(r"([0-9\.])", r" \1 ", text)
                input_ids = self.tokenizer.encode(text, add_special_tokens=False)
                types = [type_id] * len(input_ids)

                def get_dpe(tokens, number_groups):
                    number_ids = [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 119]
                    numbers = [i for i, j in enumerate(tokens) if j in number_ids]
                    numbers_cnt = 0
                    # data_dpe = [0] * len(tokens)
                    data_dpe = [0 if j == 0 else self.others_dpe_id for j in tokens]
                    for group in number_groups:
                        if group[0] == "." * len(group[0]):
                            numbers_cnt += len(group[0])
                            continue

                        start = numbers[numbers_cnt]
                        end = numbers[numbers_cnt + len(group[0]) - 1] + 1
                        corresponding_numbers = tokens[start:end]
                        digits = [i for i, j in enumerate(corresponding_numbers) if j==119]

                        # Case Integer
                        if len(digits) == 0:
                            data_dpe[start:end] = list(range(len(group[0]) + 5, 5, -1))
                        # Case Float
                        elif len(digits) == 1:
                            digit_idx = len(group[0]) - digits[0]
                            data_dpe[start:end] = list(
                                range(len(group[0]) + 5 - digit_idx, 5 - digit_idx, -1)
                            )
                        else:
                            logger.warn(f"{data_dpe[start:end]} has irregular numerical formats")

                        numbers_cnt += len(group[0])
                    return data_dpe

                dpes = get_dpe(input_ids, number_groups)
                return input_ids, types, dpes

            encoded_table_name = process_unit(table_name, self.table_type_id)
            encoded_cols = {k: process_unit(k, self.column_type_id) for k in events.columns}
 
            # Extract unique values for vocab dictionary
            vocab_dict[table_name] = {col: {'value': set(), 'word': set(), 'numeric': False} for col in events.columns if col not in [self.icustay_key, "TIME"]}
            for col in events.columns:
                if col in [self.icustay_key, "TIME"]:
                    continue
                if col in table["numeric"]:
                    # Filter out null and empty string values and calculate min and max
                    non_null_events = events.filter((F.col(col).isNotNull()) & (F.col(col) != " "))
                    non_null_events = non_null_events.withColumn(col, F.col(col).cast("float"))

                    # Aggregate the minimum and maximum values
                    min_val = non_null_events.agg(F.min(col)).collect()[0][0]
                    max_val = non_null_events.agg(F.max(col)).collect()[0][0]

                    # Check if min_val and max_val are not null, then update the vocab_dict
                    if min_val is not None and max_val is not None:
                        vocab_dict[table_name][col]['numeric'] = True
                        vocab_dict[table_name][col]['min'] = round(float(min_val), 4)
                        vocab_dict[table_name][col]['max'] = round(float(max_val), 4)
                    else:
                        # Handle the case where min_val or max_val is None (if all values are null)
                        vocab_dict[table_name][col]['numeric'] = False
                        vocab_dict[table_name][col]['min'] = None
                        vocab_dict[table_name][col]['max'] = None

                else:
                    unique_vals = events.select(col).distinct().collect()
                    for row in unique_vals:
                        # text = re.sub(r"([0-9\.])", r" \1 ", str(row[col]))
                        text = re.sub(r"\d*\.\d+", lambda x: str(round(float(x.group(0)), 4)), str(row[col]))
                        text = re.sub(r"([0-9\.])", r" \1 ", text)
                        tokenized_text = self.tokenizer.decode(self.tokenizer.encode(text)[1:-1])

                        vocab_dict[table_name][col]['value'].add(tokenized_text)

                        words = tokenized_text.split()
                        vocab_dict[table_name][col]['word'].update(words)

            schema = StructType(
                [
                    StructField("INPUTS", ArrayType(IntegerType()), False),
                    StructField("TYPES", ArrayType(IntegerType()), False),
                    StructField("DPES", ArrayType(IntegerType()), False),
                ]
            )
            def process_row(encoded_table_name, encoded_cols):
                def _process_row(row):
                    """
                    input: row (cols: icustay_id, timestamp, ...)
                    output: (input, type, dpe)
                    """
                    row = row.asDict()
                    # Should INITIALIZE with blank arrays to prevent corruption in Pyspark... Why??
                    input_ids, types, dpes = [], [], []
                    input_ids += encoded_table_name[0]
                    types += encoded_table_name[1]
                    dpes += encoded_table_name[2]
                    
                    for col, val in row.items():
                        if col in [self.icustay_key, "TIME"] or val is None:
                            continue
                        # Remove empty events
                        elif (val == "") or (val == " "):
                            continue

                        encoded_col = encoded_cols[col]
                        encoded_val = process_unit(val, self.value_type_id)
                        if len(input_ids) + len(encoded_col[0]) + len(encoded_val[0]) + 2 <= self.max_event_token_len:
                            input_ids += encoded_col[0] + encoded_val[0]
                            types += encoded_col[1] + encoded_val[1]
                            dpes += encoded_col[2] + encoded_val[2]
                        else:
                            break
                    return input_ids, types, dpes
                return F.udf(_process_row, returnType=schema)

            events = (
                events.withColumn("tmp", process_row(encoded_table_name, encoded_cols)(F.struct(*events.columns)))
                    .withColumn("INPUTS", F.col("tmp.INPUTS"))
                    .withColumn("TYPES", F.col("tmp.TYPES"))
                    .withColumn("DPES", F.col("tmp.DPES"))
                    .select(self.icustay_key, "TIME", "INPUTS", "TYPES", "DPES")
            )
            events_dfs.append(events)
        return reduce(lambda x, y: x.union(y), events_dfs), vocab_dict

    
    def make_input(self, cohorts, events, vocab_dict, spark):
        @F.pandas_udf(returnType="TIME int", functionType=F.PandasUDFType.GROUPED_MAP)
        def _make_input(events):
            # Actually, this function does not have to return anything.
            # However, return something(TIME) is required to satisfy the PySpark requirements.
            df = events.sort_values("TIME")
            flatten_cut_idx = -1
            # Consider SEP
            flatten_lens = np.cumsum(df["INPUTS"].str.len()+1).values
            event_length = len(df)

            if flatten_lens[-1] > self.max_patient_token_len-1:
                # Consider CLS token at first of the flatten input
                flatten_cut_idx = np.searchsorted(flatten_lens, flatten_lens[-1]-self.max_patient_token_len+1)
                flatten_lens = (flatten_lens - flatten_lens[flatten_cut_idx])[flatten_cut_idx+1:]
                event_length = len(flatten_lens)
            
            # Event length should not be longer than max_event_size
            event_length = min(event_length, self.max_event_size)
            df = df.iloc[-event_length:]

            if len(df)<=self.min_event_size:
                return events["TIME"].to_frame()

            # make_hi = lambda cls_id, sep_id, iterable: [[cls_id] + list(i) + [sep_id] for i in iterable]
            # make_fl = lambda cls_id, sep_id, iterable: [cls_id] + list(chain(*[list(i) + [sep_id] for i in iterable]))
            make_hi = lambda sep_id, iterable: [list(i) + [sep_id] for i in iterable]

            hi_input = make_hi(self.sep_token_id, df["INPUTS"])
            hi_type = make_hi(self.sep_type_id, df["TYPES"])
            hi_dpe = make_hi(self.others_dpe_id, df["DPES"])

            assert len(hi_input) <= self.max_event_size, hi_input
            assert all([len(i)<=self.max_event_token_len for i in hi_input]), hi_input
            # assert len(fl_input) <= self.max_patient_token_len, fl_input

            # Add padding to save as numpy array
            hi_input = np.array([np.pad(i, (0, self.max_event_token_len - len(i)), mode='constant') for i in hi_input])
            hi_type = np.array([np.pad(i, (0, self.max_event_token_len - len(i)), mode='constant') for i in hi_type])
            hi_dpe = np.array([np.pad(i, (0, self.max_event_token_len - len(i)), mode='constant') for i in hi_dpe])

            num_padding_rows = self.max_event_size - len(hi_input)
            padding_rows = np.zeros((num_padding_rows, self.max_event_token_len))

            hi_input = np.vstack([hi_input, padding_rows])
            hi_type = np.vstack([hi_type, padding_rows])
            hi_dpe = np.vstack([hi_dpe, padding_rows])

            stay_id = df[self.icustay_key].values[0]
            floored_time = floor_time(df["TIME"].values, max_time_len=self.max_time_len)

            # Create caches (cannot write to hdf5 directly with pyspark)
            data = {
                "hi": np.stack([hi_input, hi_type, hi_dpe], axis=1).astype(np.int16),
                "time": df["TIME"].values,
                "floored_time": floored_time,
            }
            with open(os.path.join(self.cache_dir, self.ehr_name, f"{stay_id}.pkl"), "wb") as f:
                pickle.dump(data, f)
            return events["TIME"].to_frame()
        
        shutil.rmtree(os.path.join(self.cache_dir, self.ehr_name), ignore_errors=True)
        os.makedirs(os.path.join(self.cache_dir, self.ehr_name), exist_ok=True)
        """
        events.groupBy(self.icustay_key).apply(_make_input).write.mode("overwrite").format("noop").save()

        logger.info("Finish Data Preprocessing. Start to write to hdf5")

        f = h5py.File(os.path.join(self.dest, f"{self.ehr_name}.h5"), "w")
        ehr_g = f.create_group("ehr")

        # Calculate statistical metrics on the event lengths
        all_event_lengths = []

        for stay_id_file in tqdm(os.listdir(os.path.join(self.cache_dir, self.ehr_name))):
            with open(os.path.join(self.cache_dir, self.ehr_name, stay_id_file), 'rb') as f:
                data = pickle.load(f)
            all_event_lengths.append(len(data["time"]))

        avg_len = int(np.mean(all_event_lengths))
        med_len = int(np.median(all_event_lengths))
        p90_len = int(np.percentile(all_event_lengths, 90))
        p95_len = int(np.percentile(all_event_lengths, 95))
        max_len = int(np.max(all_event_lengths))

        print(f"Avg length: {avg_len}, Median length: {med_len}, 90% length: {p90_len}, 95% length: {p95_len}, Max length: {max_len}")

        active_stay_ids = []

        for stay_id_file in tqdm(os.listdir(os.path.join(self.cache_dir, self.ehr_name))):
            stay_id = stay_id_file.split(".")[0]
            with open(os.path.join(self.cache_dir, self.ehr_name, stay_id_file), 'rb') as f:
                data = pickle.load(f)
            
            # Truncate the 'hi' data to the 95th percentile length and separate into different components
            truncated_hi_input = data['hi'][:p95_len, 0]
            truncated_hi_type = data['hi'][:p95_len, 1]
            truncated_hi_dpe = data['hi'][:p95_len, 2]

            truncated_hi = np.stack([truncated_hi_input, truncated_hi_type, truncated_hi_dpe], axis=1).astype(np.int16)
            truncated_time = data['time'][:p95_len]
            truncated_floored_time = data['floored_time'][:p95_len]

            # Create a group in the HDF5 file for the current stay ID
            stay_g = ehr_g.create_group(str(stay_id))

            # Create datasets for the truncated data within the group
            stay_g.create_dataset('hi', data=truncated_hi, dtype='i2', compression='lzf', shuffle=True)
            stay_g.create_dataset('time', data=truncated_time, dtype='i')
            stay_g.create_dataset('floored_time', data=truncated_floored_time, dtype='i')

            active_stay_ids.append(int(stay_id))

            # stay_g = ehr_g.create_group(str(stay_id))
            # stay_g.create_dataset('hi', data=data['hi'], dtype='i2', compression='lzf', shuffle=True)
            # stay_g.create_dataset('time', data=data['time'], dtype='i')
            # stay_g.create_dataset('floored_time', data=data['floored_time'], dtype='i')
            # active_stay_ids.append(int(stay_id))

        # Create Predef Vocab Dictionary
        """
        predef_vocab = dict()
        for table in vocab_dict.keys():
            table_key = re.sub(r"([0-9\.])", r" \1 ", table)
            table_key = self.tokenizer.decode(self.tokenizer.encode(table_key)[1:-1])
            predef_vocab[table_key] = dict()
            for column in vocab_dict[table].keys():
                column_key = re.sub(r"([0-9\.])", r" \1 ", column)
                column_key = self.tokenizer.decode(self.tokenizer.encode(column_key)[1:-1])
                numeric = vocab_dict[table][column]['numeric']

                if numeric:
                    predef_vocab[table_key][column_key] = (
                        (vocab_dict[table][column]['min'], vocab_dict[table][column]['max']),
                        numeric
                    )
                else:
                    predef_vocab[table_key][column_key] = (
                        {
                            "word": list(set(vocab_dict[table][column]['word'])),
                            "value": list(set(vocab_dict[table][column]['value']))
                        },
                        numeric
                    )

        with open(os.path.join(self.dest, f"{self.ehr_name}_predef_vocab.pickle"), 'wb') as handle:
            pickle.dump(predef_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return
        shutil.rmtree(os.path.join(self.cache_dir, self.ehr_name), ignore_errors=True)
        # Drop patients with few events

        if not isinstance(cohorts, pd.DataFrame):
            cohorts = cohorts.toPandas()
            print(cohorts)

        logger.info("Total {} patients in the cohort are skipped due to few events".format(len(cohorts) - len(active_stay_ids)))
        cohorts = cohorts[cohorts[self.icustay_key].isin(active_stay_ids)]

        # Should consider pat_id for split
        for seed in self.seed:
            shuffled = cohorts.groupby(self.patient_key)[self.patient_key].count().sample(frac=1, random_state=seed)
            cum_len = shuffled.cumsum()

            cohorts.loc[cohorts[self.patient_key].isin(
                shuffled[cum_len < int(sum(shuffled)*self.valid_percent)].index), f'split_{seed}'] = 'test'
            cohorts.loc[cohorts[self.patient_key].isin(
                shuffled[(cum_len >= int(sum(shuffled)*self.valid_percent)) 
                & (cum_len < int(sum(shuffled)*2*self.valid_percent))].index), f'split_{seed}'] = 'valid'
            cohorts.loc[cohorts[self.patient_key].isin(
                shuffled[cum_len >= int(sum(shuffled)*2*self.valid_percent)].index), f'split_{seed}'] = 'train'

        cohorts.to_csv(os.path.join(self.dest, f'{self.ehr_name}_cohort.csv'), index=False)

        # Record corhots df to hdf5
        for _, row in cohorts.iterrows():
            group = ehr_g[str(row[self.icustay_key])]
            for col in cohorts.columns:
                if col in ["INTIME", "OUTTIME"] or isinstance(row[col], (pd.Timestamp, pd.Timedelta)):
                    continue
                group.attrs[col] = row[col]
        f.close()
        logger.info("Done encoding events.")

        return

    def run_pipeline(self, spark) -> None:
        cohorts = self.build_cohorts(cached=self.cache)
        labeled_cohorts = self.prepare_tasks(cohorts, spark, cached=self.cache)
        events, vocab_dict = self.process_tables(labeled_cohorts, spark)
        self.make_input(labeled_cohorts, events, vocab_dict, spark)

    def add_special_tokens(self, new_special_tokens: Union[str, List]) -> None:
        if isinstance(new_special_tokens, str):
            new_special_tokens = [new_special_tokens]

        num_special_tokens = self.num_special_tokens
        overlapped = []
        for new_special_token in new_special_tokens:
            if new_special_token in self.special_tokens_dict:
                overlapped.append(new_special_token)

        if len(overlapped) > 0:
            logger.warn(
                "There are some tokens that have already been set to special tokens."
                " Please provide only NEW tokens. Aborted."
            )
            return None
        elif num_special_tokens + len(new_special_tokens) > self.max_special_tokens:
            logger.warn(
                f"Total additional special tokens should be less than {self.max_special_tokens}"
                " Aborted."
            )
            return None

        self.special_tokens_dict.update({
            k: "[unused{}]".format(i)
            for i, k in enumerate(new_special_tokens, start=num_special_tokens+1)
        })

    def make_compatible(self, icustays):
        """
        make different ehrs compatible with one another here
        NOTE: outtime/dischtime is converted to relative minutes from intime
            but, maintain the intime as the original value for later use
        """
        raise NotImplementedError()

    def is_compatible(self, icustays):
        checklist = [
            self.hadm_key,
            self.icustay_key,
            self.patient_key,
            "LOS",
            "AGE",
            "INTIME",
            "OUTTIME",
            "DISCHTIME",
            "IN_ICU_MORTALITY",
            "HOS_DISCHARGE_LOCATION"
        ]
        for item in checklist:
            if item not in icustays.columns.to_list():
                return False
        return True

    def save_to_cache(self, f, fname, use_pickle=False) -> None:
        if use_pickle:
            import pickle
            with open(os.path.join(self.cache_dir, fname), "wb") as fptr:
                pickle.dump(f, fptr)
        else:
            f.to_pickle(
                os.path.join(self.cache_dir, fname)
            )

    def load_from_cache(self, fname):
        cached = os.path.join(self.cache_dir, fname)
        if os.path.exists(cached):
            data = pd.read_pickle(cached)

            logger.info(
                "Loaded data from {}".format(cached)
            )
            return data
        else:
            return None

    def infer_data_extension(self) -> str:
        raise NotImplementedError()

    def download_ehr_from_url(self, url, dest) -> None:
        username = input("Email or Username: ")
        subprocess.run(
            [
                "wget", "-r", "-N", "-c", "np",
                "--user", username,
                "--ask-password", url,
                "-P", dest,
            ]
        )
        output_dir = url.replace("https://", "").replace("http://", "")

        if not os.path.exists(os.path.join(dest, output_dir)):
            raise AssertionError(
                "Download failed. Please check your network connection or "
                "if you log in with a credentialed user"
            )

    def download_ccs_from_url(self, dest) -> None:
        subprocess.run(
            [
                "wget", "-N", "-c",
                "https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip",
                "-P", dest
            ]
        )

        import zipfile

        with zipfile.ZipFile(
            os.path.join(dest, "Multi_Level_CCS_2015.zip"), "r"
        ) as zip_ref:
            zip_ref.extractall(os.path.join(dest, "foo.d"))
        os.rename(
            os.path.join(dest, "foo.d", "ccs_multi_dx_tool_2015.csv"),
            os.path.join(dest, "ccs_multi_dx_tool_2015.csv")
        )
        os.remove(os.path.join(dest, "Multi_Level_CCS_2015.zip"))
        shutil.rmtree(os.path.join(dest, "foo.d"))

    def download_icdgem_from_url(self, dest) -> None:
        subprocess.run(
            [
                "wget", "-N", "-c",
                "https://data.nber.org/gem/icd10cmtoicd9gem.csv",
                "-P", dest,
            ]
        )


def floor_time(data, time_window=10, max_time_len=2):
    """
    Floors the input data to the nearest time window and optionally scales it down.
    
    Parameters:
    - data (array-like): The input time data to be floored.
    - time_window (int): The time window to floor the data to.
    - max_time_len (int): Determines the format of the returned data. If 3, data is split into
      hundreds, tens, and ones places. If 2 (default), data is split into tens and ones.

    Returns:
    - list: A list of floored (and optionally scaled) time data, split according to max_time_len.
    """
    
    # Floor the data to the nearest time window
    floored_data = np.floor(data / time_window).astype(int) * time_window
    # Scale down the data for further processing
    scaled_down_data = floored_data // 10
    
    # Split the scaled data based on max_time_len
    if max_time_len == 3:
        return np.array([[x // 100, (x % 100) // 10, x % 10] for x in scaled_down_data])
    else:
        return np.array([[x // 10, x % 10] for x in scaled_down_data])