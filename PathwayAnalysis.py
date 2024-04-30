import requests
import regex as re
import random
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

        self.pathway_pairs_sc = self.__PIRM_accuracy(self.severe, self.control, self.pathway_pairs)
        self.non_pathway_pairs_sc = self.__PIRM_accuracy(self.severe, self.control, self.non_pathway_pairs)
        self.pathway_pairs_sm = self.__PIRM_accuracy(self.severe, self.mild, self.pathway_pairs)
        self.non_pathway_pairs_sm = self.__PIRM_accuracy(self.severe, self.mild, self.non_pathway_pairs)
        self.pathway_pairs_mc = self.__PIRM_accuracy(self.mild, self.control, self.pathway_pairs)
        self.non_pathway_pairs_mc = self.__PIRM_accuracy(self.mild, self.control, self.non_pathway_pairs)

        self.df = pd.DataFrame(
            {
                "pair": self.pathway_pairs + self.non_pathway_pairs,
                "mcc_sc": list(self.pathway_pairs_sc["mcc"]) + list(self.non_pathway_pairs_sc["mcc"]),
                "f1_sc": list(self.pathway_pairs_sc["f1"]) + list(self.non_pathway_pairs_sc["f1"]),
                "spec_sc": list(self.pathway_pairs_sc["spec"]) + list(self.non_pathway_pairs_sc["spec"]),
                "sens_sc": list(self.pathway_pairs_sc["sens"]) + list(self.non_pathway_pairs_sc["sens"]),
                "mcc_mc": list(self.pathway_pairs_mc["mcc"]) + list(self.non_pathway_pairs_mc["mcc"]),
                "f1_mc": list(self.pathway_pairs_mc["f1"]) + list(self.non_pathway_pairs_mc["f1"]),
                "spec_mc": list(self.pathway_pairs_mc["spec"]) + list(self.non_pathway_pairs_mc["spec"]),
                "sens_mc": list(self.pathway_pairs_mc["sens"]) + list(self.non_pathway_pairs_mc["sens"]),
                "mcc_sm": list(self.pathway_pairs_sm["mcc"]) + list(self.non_pathway_pairs_sm["mcc"]),
                "f1_sm": list(self.pathway_pairs_sm["f1"]) + list(self.non_pathway_pairs_sm["f1"]),
                "spec_sm": list(self.pathway_pairs_sm["spec"]) + list(self.non_pathway_pairs_sm["spec"]),
                "sens_sm": list(self.pathway_pairs_sm["sens"]) + list(self.non_pathway_pairs_sm["sens"]),
                "pathways": [1] * len(self.pathway_pairs) + [0] * len(self.non_pathway_pairs),
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
        self.pathway_pairs = []
        self.num_pathways = []
        self.non_pathway_pairs = []
        self.all_pathways = []
        pairs = list(combinations(self.genes, 2))
        for pair in tqdm(pairs):
            pathways = self.__pathways_in_common(pair)
            if pathways != -1 and len(pathways):
                self.pathway_pairs.append(pair)
                self.num_pathways.append(len(pathways))
                self.all_pathways += pathways
            elif self.__is_interactions(pair):
                self.pathway_pairs.append(pair)
                self.num_pathways.append(0)
            else:
                self.non_pathway_pairs.append(pair)
        self.all_pathways = list(set(self.all_pathways))
        self.pathway_dict = self.__make_pathway_dict()
        with open("pairs.pkl", "wb") as f:
            pickle.dump((self.pathway_pairs, self.non_pathway_pairs, self.num_pathways, self.all_pathways, self.pathway_dict), f)

    def __load_pairs(self, p):
        with open(p, "rb") as f:
            self.pathway_pairs, self.non_pathway_pairs, self.num_pathways, self.all_pathways, self.pathway_dict = pickle.load(f)

    def __pathways_in_common(self, pair):
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

    def __make_pathway_dict(self):
        d = {key: [] for key in self.all_pathways}
        
        for pair in self.pathway_pairs:
            p = self.__pathways_in_common(pair)
            if p != -1:
                for pathway in p:
                    if pair[0] not in d[pathway]:
                        d[pathway].append(pair[0])
                    if pair[1] not in d[pathway]:
                        d[pathway].append(pair[1])
        return d
    
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
            axes[0, 0], "Severe vs Control (sens)", self.pathway_pairs_sc[0], self.non_pathway_pairs_sc[0]
        )
        self.__hist_axis(
            axes[0, 1], "Mild vs Control (sens)", self.pathway_pairs_mc[0], self.non_pathway_pairs_mc[0]
        )
        self.__hist_axis(
            axes[0, 2], "Severe vs Mild (sens)", self.pathway_pairs_sm[0], self.non_pathway_pairs_sm[0]
        )

        self.__hist_axis(
            axes[1, 0], "Severe vs Control (spec)", self.pathway_pairs_sc[1], self.non_pathway_pairs_sc[1]
        )
        self.__hist_axis(
            axes[1, 1], "Mild vs Control (spec)", self.pathway_pairs_mc[1], self.non_pathway_pairs_mc[1]
        )
        self.__hist_axis(
            axes[1, 2], "Severe vs Mild (spec)", self.pathway_pairs_sm[1], self.non_pathway_pairs_sm[1]
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
                'mild_SSres':[mild_SSres.loc[pair[0], pair[1]] for pair in self.non_pathway_pairs],
                'mild_R2':[mild_R2.loc[pair[0], pair[1]] for pair in self.non_pathway_pairs],
                'severe_SSres':[severe_SSres.loc[pair[0], pair[1]] for pair in self.non_pathway_pairs],
                'severe_R2':[severe_R2.loc[pair[0], pair[1]] for pair in self.non_pathway_pairs],
                'control_SSres':[control_SSres.loc[pair[0], pair[1]] for pair in self.non_pathway_pairs],
                'control_R2':[control_R2.loc[pair[0], pair[1]] for pair in self.non_pathway_pairs]
            },
            index=self.non_pathway_pairs
        )
        pathway = pd.DataFrame(
            {
                'mild_SSres':[mild_SSres.loc[pair[0], pair[1]] for pair in self.pathway_pairs],
                'mild_R2':[mild_R2.loc[pair[0], pair[1]] for pair in self.pathway_pairs],
                'severe_SSres':[severe_SSres.loc[pair[0], pair[1]] for pair in self.pathway_pairs],
                'severe_R2':[severe_R2.loc[pair[0], pair[1]] for pair in self.pathway_pairs],
                'control_SSres':[control_SSres.loc[pair[0], pair[1]] for pair in self.pathway_pairs],
                'control_R2':[control_R2.loc[pair[0], pair[1]] for pair in self.pathway_pairs]
            },
            index=self.pathway_pairs
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
    
    def accuracies_for_gene(self, genes = None, pathway = None):
        _, axs = plt.subplots(4, 3, figsize=(20, 20))
        if genes == None:
            print(pathway, self.pathway_dict[pathway])
            genes = self.pathway_dict[pathway]
        for i, col in enumerate(self.df.columns[1:-1]):
            rows_containing_gene = self.df.loc[self.df['pair'].astype(str).str.contains('|'.join(genes))]
            axs[i%4,i//4].boxplot([rows_containing_gene[col], self.df[col]], labels=[f'containing select genes', f'all accuracies'])
            axs[i%4,i//4].set_title(col)
        plt.show()
    
    def get_pathway_pair(self):
        gene_list = random.choice(list(self.pathway_dict.values()))
        pair = random.sample(gene_list, 2)
        sc = self.__PIRM_accuracy(self.severe, self.control, [tuple(pair)])
        mc = self.__PIRM_accuracy(self.mild, self.control, [tuple(pair)])
        sm = self.__PIRM_accuracy(self.severe, self.mild, [tuple(pair)])
        df =  pd.concat([sc, mc, sm])
        df.index = ['sc' ,'mc', 'sm']
        return df
    
    def get_random_pair(self):
        two_pathways = random.sample(sorted(self.pathway_dict.values()), 2)
        gene1 = random.choice(two_pathways[0])
        gene2 = random.choice([x for x in two_pathways[1] if x != gene1])
        sc = self.__PIRM_accuracy(self.severe, self.control, [(gene1, gene2)])
        mc = self.__PIRM_accuracy(self.mild, self.control, [(gene1, gene2)])
        sm = self.__PIRM_accuracy(self.severe, self.mild, [(gene1, gene2)])
        df =  pd.concat([sc, mc, sm])
        df.index = ['sc' ,'mc', 'sm']
        return df