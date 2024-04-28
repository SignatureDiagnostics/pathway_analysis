import requests
import regex as re
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations
import pickle
from regression import Regressions
from PIRM.PIRMDiscovery import CT_From_Table, DiscoverySearch
import matplotlib.pyplot as plt


class PathwayAnalysis:
    def __init__(
        self,
        interactions_path,
        gene_path,
        rnacounts_path,
        metadata_path,
        pickle_path=None,
    ):
        self.interactions = open(interactions_path, "r").readlines()
        self.to_name = pd.read_csv(gene_path).set_index("code")["name"].to_dict()
        self.to_ensembl = pd.read_csv(gene_path).set_index("name")["code"].to_dict()

        self.genes = [
            v[1]
            for v in self.to_name.items()
            if v[0] in pd.read_csv(rnacounts_path)["Unnamed: 0"].tolist()
        ]
        self.codes = [
            v[0]
            for v in self.to_name.items()
            if v[0] in pd.read_csv(rnacounts_path)["Unnamed: 0"].tolist()
        ]
        self.counts = np.log1p(pd.read_csv(rnacounts_path, index_col=0)).loc[
            self.codes, :
        ]
        self.meta = pd.read_csv(
            metadata_path, header=None, names=["sample", "classification"]
        )
        self.mild = self.counts[self.meta.loc[self.meta["classification"] == "Mild", "sample"].tolist()]
        self.mild.index = self.mild.index.map(self.to_name)
        self.severe = self.counts[self.meta.loc[self.meta["classification"] == "Severe", "sample"].tolist()]
        self.severe.index = self.severe.index.map(self.to_name)
        self.control = self.counts[self.meta.loc[self.meta["classification"] == "Control", "sample"].tolist()]
        self.control.index = self.control.index.map(self.to_name)

        if pickle_path:
            self.__load_pairs(pickle_path)
        else:
            print("No pickle file passed: creating pairs")
            self.__sort_into_pairs()

        self.A_sc = self.__PIRM_accuracy(self.severe, self.control, self.A)
        self.B_sc = self.__PIRM_accuracy(self.severe, self.control, self.B)
        self.A_sm = self.__PIRM_accuracy(self.severe, self.mild, self.A)
        self.B_sm = self.__PIRM_accuracy(self.severe, self.mild, self.B)
        self.A_mc = self.__PIRM_accuracy(self.mild, self.control, self.A)
        self.B_mc = self.__PIRM_accuracy(self.mild, self.control, self.B)

        self.df = pd.DataFrame(
            {
                "pair": self.A + self.B,
                "mcc_sc": list(self.A_sc["mcc"]) + list(self.B_sc["mcc"]),
                "f1_sc": list(self.A_sc["f1"]) + list(self.B_sc["f1"]),
                "spec_sc": list(self.A_sc["spec"]) + list(self.B_sc["spec"]),
                "sens_sc": list(self.A_sc["sens"]) + list(self.B_sc["sens"]),
                "mcc_mc": list(self.A_mc["mcc"]) + list(self.B_mc["mcc"]),
                "f1_mc": list(self.A_mc["f1"]) + list(self.B_mc["f1"]),
                "spec_mc": list(self.A_mc["spec"]) + list(self.B_mc["spec"]),
                "sens_mc": list(self.A_mc["sens"]) + list(self.B_mc["sens"]),
                "mcc_sm": list(self.A_sm["mcc"]) + list(self.B_sm["mcc"]),
                "f1_sm": list(self.A_sm["f1"]) + list(self.B_sm["f1"]),
                "spec_sm": list(self.A_sm["spec"]) + list(self.B_sm["spec"]),
                "sens_sm": list(self.A_sm["sens"]) + list(self.B_sm["sens"]),
                "pathways": [1] * len(self.A) + [0] * len(self.B),
            }
        )

    def __sort_into_pairs(self):
        """Creates three lists: A, which is the
        list of all pairs that share at least one pathway or interaction,
        B, which is the list of pairs that do not share a pathway, and
        num_pathways which lists the number of shared pathways for each value
        in A. If there are no pathways, only other interactions, it lists 0
        pathways. It then saves the data as a pickle because it takes a long
        time to generate
        """
        A = []
        num_pathways = []
        B = []
        pairs = list(combinations(self.genes, 2))
        for pair in tqdm(pairs):
            pathways = self.pathways_in_common(pair)
            if pathways != -1 and len(pathways):
                A.append(pair)
                num_pathways.append(len(pathways))
            elif self.__is_interactions(pair):
                A.append(pair)
                num_pathways.append(0)
            else:
                B.append(pair)
        with open("A_B.pkl", "wb") as f:
            pickle.dump((A, B, num_pathways), f)

    def __load_pairs(self, p):
        with open(p, "rb") as f:
            self.A, self.B, self.num_pathways = pickle.load(f)

    def pathways_in_common(self, pair):
        gene1, gene2 = pair
        pattern = r'"name":"(.*?)","dataSource":'
        gene_1_pathways = re.findall(
            pattern,
            requests.get(
                "https://www.pathwaycommons.org/pc2/top_pathways?q=" + gene1
            ).text,
        )
        gene_2_pathways = re.findall(
            pattern,
            requests.get(
                "https://www.pathwaycommons.org/pc2/top_pathways?q=" + gene2
            ).text,
        )

        if gene_1_pathways == [] and gene_2_pathways == []:
            return -1
        return [value for value in gene_1_pathways if value in gene_2_pathways]

    def __is_interactions(self, pair):
        for i in self.interactions:
            if pair[0] in i and pair[1] in i:
                return True
        return False

    def __PIRM_accuracy(self, case, control, pairs):
        ct_case = CT_From_Table(
            sample_ids=case.columns, group_label="case", table=case.T
        )
        ct_ctrl = CT_From_Table(
            sample_ids=control.columns, group_label="control", table=control.T
        )
        d = DiscoverySearch(ct_case=ct_case, ct_ctrl=ct_ctrl)
        d.test_all_combos()
        results = pd.DataFrame(
            d.results,
            columns=[
                "gene1",
                "gene2",
                "mcc",
                "f1",
                "sens",
                "spec",
                "tp",
                "fn",
                "tn",
                "fp",
            ],
        )
        results["pair"] = list(zip(results["gene1"], results["gene2"])
        )
        results = results.drop(columns=["gene1", "gene2"])
        results["pair"] = pd.Categorical(
            results["pair"], categories=pairs, ordered=True
        )
        results = results.sort_values(by="pair")
        results = results.dropna(subset=["pair"])
        return results
    
    def __get_search_cols(self, contrast, metric):
        search_col_map = {
            "mild/severe": {
                "spec": ["spec_sm"],
                "sens": ["sens_sm"],
                "mcc": ["mcc_sm"],
                "f1": ["f1_sm"],
                None: ["mcc_sm", "f1_sm", "sens_sm", "spec_sm"],
            },
            "mild/control": {
                "spec": ["spec_mc"],
                "sens": ["sens_mc"],
                "mcc": ["mcc_mc"],
                "f1": ["f1_mc"],
                None: ["mcc_mc", "f1_mc", "sens_mc", "spec_mc"],
            },
            "severe/control": {
                "spec": ["spec_sc"],
                "sens": ["sens_sc"],
                "mcc": ["mcc_sc"],
                "f1": ["f1_sc"],
                None: ["mcc_sc", "f1_sc", "sens_sc", "spec_sc"],
            },
            None: {
                "spec": ["spec_sc", "spec_mc", "spec_sm"],
                "sens": ["sens_sc", "sens_mc", "sens_sm"],
                "mcc": ["mcc_sc", "mcc_mc", "mcc_sm"],
                "f1": ["f1_sc", "f1_mc", "f1_sm"],
                None: [
                    "spec_sc",
                    "spec_mc",
                    "spec_sm",
                    "sens_sc",
                    "sens_mc",
                    "sens_sm",
                    "mcc_sc",
                    "mcc_mc",
                    "mcc_sm",
                    "f1_sc",
                    "f1_mc",
                    "f1_sm",
                ],
            },
        }
        return search_col_map.get(contrast, {}).get(metric, "error")

    def __hist_axis(self, ax, name, A, B):
        bins = np.arange(0, 1, 0.1)
        a = np.histogram(A, bins=bins)
        b = np.histogram(B, bins=bins)
        ax.bar(
            bins[:-1],
            (a[0] / sum(a[0])) / (b[0] / sum(b[0])),
            width=0.09,
            color="black",
        )
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.axhline(1)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("pathway density over non-pathway density")

    def hist(self):
        _, axes = plt.subplots(2, 3, figsize=(20, 10))
        self.__hist_axis(
            axes[0, 0], "Severe vs Control (sens)", self.A_sc[0], self.B_sc[0]
        )
        self.__hist_axis(
            axes[0, 1], "Mild vs Control (sens)", self.A_mc[0], self.B_mc[0]
        )
        self.__hist_axis(
            axes[0, 2], "Severe vs Mild (sens)", self.A_sm[0], self.B_sm[0]
        )

        self.__hist_axis(
            axes[1, 0], "Severe vs Control (spec)", self.A_sc[1], self.B_sc[1]
        )
        self.__hist_axis(
            axes[1, 1], "Mild vs Control (spec)", self.A_mc[1], self.B_mc[1]
        )
        self.__hist_axis(
            axes[1, 2], "Severe vs Mild (spec)", self.A_sm[1], self.B_sm[1]
        )
        plt.savefig("figures/histograms.png")
        plt.show()

    def get(self, range: tuple, pathways=None, contrast=None, metric=None):
        
        mask = self.df[self.__get_search_cols(contrast, metric)].apply(
            lambda row: any(range[0] <= x <= range[1] for x in row), axis=1
        )
        filtered_df = self.df[mask]
        if pathways == True:
            return list(filtered_df[filtered_df["pathways"] == 1]["pair"])
        elif pathways == False:
            return list(filtered_df[filtered_df["pathways"] == 0]["pair"])

        return list(filtered_df["pair"])

    def prob_pathways(self, range: tuple, contrast=None, metric=None):
        pathways = len(self.get(range, pathways=True, contrast=contrast, metric=metric))
        no_pathways = len(
            self.get(range, pathways=False, contrast=contrast, metric=metric)
        )
        try:
            return pathways / (pathways + no_pathways)
        except:
            return 0

    def get_regression(self):
        mild_reg = Regressions(X=np.array(self.mild).T)
        severe_reg = Regressions(X=np.array(self.severe).T)
        control_reg = Regressions(X=np.array(self.control).T)
        mild_SSres, mild_R2, severe_SSres, severe_R2, control_SSres, control_R2 = (
            mild_reg.SSres,
            mild_reg.R2,
            severe_reg.SSres,
            severe_reg.R2,
            control_reg.SSres,
            control_reg.R2,
        )
        mild_SSres = pd.DataFrame(mild_SSres, columns=self.mild.index, index=self.mild.index)
        mild_R2 = pd.DataFrame(mild_R2, columns=self.mild.index, index=self.mild.index)
        severe_SSres = pd.DataFrame(severe_SSres, columns=self.severe.index, index=self.severe.index)
        severe_R2 = pd.DataFrame(severe_R2, columns=self.severe.index, index=self.severe.index)
        control_SSres = pd.DataFrame(control_SSres, columns=self.control.index, index=self.control.index)
        control_R2 = pd.DataFrame(control_R2, columns=self.control.index, index=self.control.index)

        no_pathway = pd.DataFrame(
            {
                'mild_SSres':[mild_SSres.loc[pair[0], pair[1]] for pair in self.B],
                'mild_R2':[mild_R2.loc[pair[0], pair[1]] for pair in self.B],
                'severe_SSres':[severe_SSres.loc[pair[0], pair[1]] for pair in self.B],
                'severe_R2':[severe_R2.loc[pair[0], pair[1]] for pair in self.B],
                'control_SSres':[control_SSres.loc[pair[0], pair[1]] for pair in self.B],
                'control_R2':[control_R2.loc[pair[0], pair[1]] for pair in self.B]
            },
            index=self.B
        )
        pathway = pd.DataFrame(
            {
                'mild_SSres':[mild_SSres.loc[pair[0], pair[1]] for pair in self.A],
                'mild_R2':[mild_R2.loc[pair[0], pair[1]] for pair in self.A],
                'severe_SSres':[severe_SSres.loc[pair[0], pair[1]] for pair in self.A],
                'severe_R2':[severe_R2.loc[pair[0], pair[1]] for pair in self.A],
                'control_SSres':[control_SSres.loc[pair[0], pair[1]] for pair in self.A],
                'control_R2':[control_R2.loc[pair[0], pair[1]] for pair in self.A]
            },
            index=self.A
        )

        return no_pathway, pathway
    
    def contingency(self, comparison, metric,):
        col = self.__get_search_cols(comparison, metric)[0]
        col_pathways = pd.concat([self.df[col], self.df['pathways']], axis=1)
        a = len(col_pathways[(col_pathways[col] >= 0.5) & (col_pathways['pathways'] == 1)])
        b = len(col_pathways[(col_pathways[col] >= 0.5) & (col_pathways['pathways'] == 0)])
        c = len(col_pathways[(col_pathways[col] < 0.5) & (col_pathways['pathways'] == 1)])
        d = len(col_pathways[(col_pathways[col] < 0.5) & (col_pathways['pathways'] == 0)])
        return pd.DataFrame({'>= 0.5': [a,b], '< 0.5': [c,d]}, index=['pathways','no pathways'])
    
    def accuracies_for_gene(self, gene):
        fig, axs = plt.subplots(4, 3, figsize=(20, 20))
        for i, col in enumerate(self.df.columns[1:-1]):
            rows_containing_gene = self.df.loc[self.df['pair'].astype(str).str.contains(gene)]
            axs[i%4,i//4].boxplot([rows_containing_gene[col], self.df[col]], labels=[f'containing {gene}', f'all accuracies'])
            axs[i%4,i//4].set_title(col)
        plt.show()