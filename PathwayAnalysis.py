import requests
import regex as re
import pandas as pd
import numpy as np
np.random.seed(42)
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
from regression import Regressions
from PIRM.PIRMDiscovery import CT_From_Table, DiscoverySearch
import matplotlib.pyplot as plt


class PathwayAnalysis:
    def __init__(self, gene_path, rnacounts_path, metadata_path):
        self.to_name = pd.read_csv(gene_path).set_index("code")["name"].to_dict()
        self.to_ensembl = pd.read_csv(gene_path).set_index("name")["code"].to_dict()
        unfiltered_counts = np.log1p(pd.read_csv(rnacounts_path, index_col=0))
        codes = [g for g in list(pd.read_csv(gene_path)["code"]) if g in unfiltered_counts.index]
        self.counts = unfiltered_counts.loc[list(set(codes))]
        self.genes = list(set([self.to_name[x] for x in codes]))
        self.meta = pd.read_csv(metadata_path, header=None, names=["sample", "class"])

        self.mild = self.counts[self.meta.loc[self.meta["class"] == "Mild", "sample"].tolist()]
        self.mild.index = self.mild.index.map(self.to_name)
        self.severe = self.counts[self.meta.loc[self.meta["class"] == "Severe", "sample"].tolist()]
        self.severe.index = self.severe.index.map(self.to_name)
        self.control = self.counts[self.meta.loc[self.meta["class"] == "Control", "sample"].tolist()]
        self.control.index = self.control.index.map(self.to_name)

        d = {}
        for g in tqdm(self.genes):
            d[g] = re.findall(
                r'"name":"(.*?)","dataSource":',
                requests.get(
                    "https://www.pathwaycommons.org/pc2/top_pathways?q=" + g
                ).text,
            )

        self.d = d

        self.pathways_to_genes = defaultdict(list)
        for gene, pathways in d.items():
            for pathway in pathways:
                self.pathways_to_genes[pathway].append(gene)
        self.pathways_to_genes = dict(self.pathways_to_genes)

        self.pathway_pairs = []
        self.non_pathway_pairs = []

        key_pairs = combinations(d.keys(), 2)
        for key_pair in key_pairs:
            key1, key2 = key_pair
            if bool(set(d[key1]) & set(d[key2])):
                self.pathway_pairs.append((key1, key2))
            else:
                self.non_pathway_pairs.append((key1, key2))

        self.pathway_pairs_sc = self.__PIRM_accuracy(
            self.severe, self.control, self.pathway_pairs
        )
        self.non_pathway_pairs_sc = self.__PIRM_accuracy(
            self.severe, self.control, self.non_pathway_pairs
        )
        self.pathway_pairs_sm = self.__PIRM_accuracy(
            self.severe, self.mild, self.pathway_pairs
        )
        self.non_pathway_pairs_sm = self.__PIRM_accuracy(
            self.severe, self.mild, self.non_pathway_pairs
        )
        self.pathway_pairs_mc = self.__PIRM_accuracy(
            self.mild, self.control, self.pathway_pairs
        )
        self.non_pathway_pairs_mc = self.__PIRM_accuracy(
            self.mild, self.control, self.non_pathway_pairs
        )

        self.df = pd.DataFrame(
            {
                "pair": self.pathway_pairs + self.non_pathway_pairs,
                "mcc_sc": list(self.pathway_pairs_sc["mcc"])
                + list(self.non_pathway_pairs_sc["mcc"]),
                "f1_sc": list(self.pathway_pairs_sc["f1"])
                + list(self.non_pathway_pairs_sc["f1"]),
                "spec_sc": list(self.pathway_pairs_sc["spec"])
                + list(self.non_pathway_pairs_sc["spec"]),
                "sens_sc": list(self.pathway_pairs_sc["sens"])
                + list(self.non_pathway_pairs_sc["sens"]),
                "mcc_mc": list(self.pathway_pairs_mc["mcc"])
                + list(self.non_pathway_pairs_mc["mcc"]),
                "f1_mc": list(self.pathway_pairs_mc["f1"])
                + list(self.non_pathway_pairs_mc["f1"]),
                "spec_mc": list(self.pathway_pairs_mc["spec"])
                + list(self.non_pathway_pairs_mc["spec"]),
                "sens_mc": list(self.pathway_pairs_mc["sens"])
                + list(self.non_pathway_pairs_mc["sens"]),
                "mcc_sm": list(self.pathway_pairs_sm["mcc"])
                + list(self.non_pathway_pairs_sm["mcc"]),
                "f1_sm": list(self.pathway_pairs_sm["f1"])
                + list(self.non_pathway_pairs_sm["f1"]),
                "spec_sm": list(self.pathway_pairs_sm["spec"])
                + list(self.non_pathway_pairs_sm["spec"]),
                "sens_sm": list(self.pathway_pairs_sm["sens"])
                + list(self.non_pathway_pairs_sm["sens"]),
                "pathways": [1] * len(self.pathway_pairs)
                + [0] * len(self.non_pathway_pairs),
            }
        )

    def __PIRM_accuracy(self, case, control, pairs):
        def select_non_nan(row):
            if not pd.isna(row['pair1']):
                return row['pair1']
            elif not pd.isna(row['pair2']):
                return row['pair2']
            else:
                return np.nan
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

        #in order to sort the dataframe you have to make two pair items
        # because the tuples are not in order
        results["pair1"] = list(zip(results["gene1"], results["gene2"]))
        results["pair2"] = list(zip(results["gene2"], results["gene1"]))
        results = results.drop(columns=["gene1", "gene2"])

        results["pair1"] = pd.Categorical(
            results["pair1"], categories=pairs, ordered=True
        )

        results["pair2"] = pd.Categorical(
            results["pair2"], categories=pairs, ordered=True
        )
        results['pair'] = results.apply(select_non_nan, axis=1)
        results = results.drop(columns=["pair1", "pair2"])

        results["pair"] = pd.Categorical(
            results["pair"], categories=pairs, ordered=True
        )
        results = results.sort_values(by="pair").drop_duplicates('pair', keep='last')
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
            axes[0, 0],
            "Severe vs Control (sens)",
            self.pathway_pairs_sc[0],
            self.non_pathway_pairs_sc[0],
        )
        self.__hist_axis(
            axes[0, 1],
            "Mild vs Control (sens)",
            self.pathway_pairs_mc[0],
            self.non_pathway_pairs_mc[0],
        )
        self.__hist_axis(
            axes[0, 2],
            "Severe vs Mild (sens)",
            self.pathway_pairs_sm[0],
            self.non_pathway_pairs_sm[0],
        )

        self.__hist_axis(
            axes[1, 0],
            "Severe vs Control (spec)",
            self.pathway_pairs_sc[1],
            self.non_pathway_pairs_sc[1],
        )
        self.__hist_axis(
            axes[1, 1],
            "Mild vs Control (spec)",
            self.pathway_pairs_mc[1],
            self.non_pathway_pairs_mc[1],
        )
        self.__hist_axis(
            axes[1, 2],
            "Severe vs Mild (spec)",
            self.pathway_pairs_sm[1],
            self.non_pathway_pairs_sm[1],
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
        mild_SSres = pd.DataFrame(
            mild_SSres, columns=self.mild.index, index=self.mild.index
        )
        mild_R2 = pd.DataFrame(mild_R2, columns=self.mild.index, index=self.mild.index)
        severe_SSres = pd.DataFrame(
            severe_SSres, columns=self.severe.index, index=self.severe.index
        )
        severe_R2 = pd.DataFrame(
            severe_R2, columns=self.severe.index, index=self.severe.index
        )
        control_SSres = pd.DataFrame(
            control_SSres, columns=self.control.index, index=self.control.index
        )
        control_R2 = pd.DataFrame(
            control_R2, columns=self.control.index, index=self.control.index
        )

        no_pathway = pd.DataFrame(
            {
                "mild_SSres": [
                    mild_SSres.loc[pair[0], pair[1]] for pair in self.non_pathway_pairs
                ],
                "mild_R2": [
                    mild_R2.loc[pair[0], pair[1]] for pair in self.non_pathway_pairs
                ],
                "severe_SSres": [
                    severe_SSres.loc[pair[0], pair[1]]
                    for pair in self.non_pathway_pairs
                ],
                "severe_R2": [
                    severe_R2.loc[pair[0], pair[1]] for pair in self.non_pathway_pairs
                ],
                "control_SSres": [
                    control_SSres.loc[pair[0], pair[1]]
                    for pair in self.non_pathway_pairs
                ],
                "control_R2": [
                    control_R2.loc[pair[0], pair[1]] for pair in self.non_pathway_pairs
                ],
            },
            index=self.non_pathway_pairs,
        )
        pathway = pd.DataFrame(
            {
                "mild_SSres": [
                    mild_SSres.loc[pair[0], pair[1]] for pair in self.pathway_pairs
                ],
                "mild_R2": [
                    mild_R2.loc[pair[0], pair[1]] for pair in self.pathway_pairs
                ],
                "severe_SSres": [
                    severe_SSres.loc[pair[0], pair[1]] for pair in self.pathway_pairs
                ],
                "severe_R2": [
                    severe_R2.loc[pair[0], pair[1]] for pair in self.pathway_pairs
                ],
                "control_SSres": [
                    control_SSres.loc[pair[0], pair[1]] for pair in self.pathway_pairs
                ],
                "control_R2": [
                    control_R2.loc[pair[0], pair[1]] for pair in self.pathway_pairs
                ],
            },
            index=self.pathway_pairs,
        )

        return no_pathway, pathway

    def contingency(
        self,
        comparison,
        metric,
    ):
        col = self.__get_search_cols(comparison, metric)[0]
        col_pathways = pd.concat([self.df[col], self.df["pathways"]], axis=1)
        a = len(
            col_pathways[(col_pathways[col] >= 0.5) & (col_pathways["pathways"] == 1)]
        )
        b = len(
            col_pathways[(col_pathways[col] >= 0.5) & (col_pathways["pathways"] == 0)]
        )
        c = len(
            col_pathways[(col_pathways[col] < 0.5) & (col_pathways["pathways"] == 1)]
        )
        d = len(
            col_pathways[(col_pathways[col] < 0.5) & (col_pathways["pathways"] == 0)]
        )
        return pd.DataFrame(
            {">= 0.5": [a, b], "< 0.5": [c, d]}, index=["pathways", "no pathways"]
        )

    def accuracies_for_gene(self, genes=None, pathway=None):
        _, axs = plt.subplots(4, 3, figsize=(20, 20))
        if genes == None:
            print(pathway, self.pathways_to_genes[pathway])
            genes = self.pathways_to_genes[pathway]
        for i, col in enumerate(self.df.columns[1:-1]):
            rows_containing_gene = self.df.loc[
                self.df["pair"].astype(str).str.contains("|".join(genes))
            ]
            axs[i % 4, i // 4].boxplot(
                [rows_containing_gene[col], self.df[col]],
                labels=[f"containing select genes", f"all accuracies"],
            )
            axs[i % 4, i // 4].set_title(col)
        plt.show()