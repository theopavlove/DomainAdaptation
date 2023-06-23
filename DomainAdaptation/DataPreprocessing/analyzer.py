import functools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as ps

from pathlib import Path
from typing import Union

method_to_symbol = {
    "adda": "square",
    "afn": "diamond",
    "cdan": "cross",
    "dan": "x",
    "dann": "triangle-up",
    "jan": "triangle-down",
    "mcc": "star",
    "mcd": "triangle-left",
    "mdd": "triangle-right",
    "src_only": "circle",
}
method_to_color = {
    "adda": "#7BD7F1",
    "afn": "#E27862",
    "cdan": "#F7D57E",
    "dan": "#75D1A9",
    "dann": "#F086A3",
    "jan": "#CAEB9E",
    "mcc": "#AF7DF4",
    "mcd": "#F3ABFA",
    "mdd": "#F4B37E",
    "src_only": "black",
}


class Analyzer:
    def add_metric_val(
        self,
        metrics: dict,
        dataset: str,
        method: str,
        metric_name: str,
        metric_val: float,
    ) -> None:
        method = method.split("-")[0].split("/")[-1]
        if dataset not in metrics:
            metrics[dataset] = {}
        if method not in metrics[dataset]:
            metrics[dataset][method] = {}
        if metric_name not in metrics[dataset][method]:
            metrics[dataset][method][metric_name] = []
        metrics[dataset][method][metric_name].append(metric_val)

    def parse_test_logs(self, path_to_logs: Union[Path, str]) -> pd.DataFrame:
        path_to_logs = Path(path_to_logs)
        total_dfs: dict = {}

        for directory in sorted(path_to_logs.iterdir()):
            test_log_ptrn = directory.glob("test-*.txt")
            path_to_test_logs = list(test_log_ptrn)[0]

            metrics: dict = {}
            train_params: dict = {}

            with open(path_to_test_logs) as test_in:
                for line in test_in:
                    if line.startswith("Namespace("):
                        line = line.replace("Namespace", "dict")
                        train_params = eval(line)  # nosec
                    else:
                        if line.startswith(" * Acc@1"):
                            acc = float(line.split()[-1])
                            self.add_metric_val(
                                metrics,
                                train_params["data_name"],
                                train_params["log"],
                                "accuracy",
                                acc,
                            )
                        elif line.startswith(" * Acc1"):
                            acc = float(line.split()[2])
                            self.add_metric_val(
                                metrics,
                                train_params["data_name"],
                                train_params["log"],
                                "accuracy",
                                acc,
                            )
                        elif line.startswith("PR AUC") or line.startswith("F1 PR AUC"):
                            pr_auc = float(line.split()[-1])
                            self.add_metric_val(
                                metrics,
                                train_params["data_name"],
                                train_params["log"],
                                "pr_auc",
                                pr_auc,
                            )
                        elif line.startswith("ROC AUC") or line.startswith(
                            "F1 ROC AUC"
                        ):
                            roc_auc = float(line.split()[-1])
                            self.add_metric_val(
                                metrics,
                                train_params["data_name"],
                                train_params["log"],
                                "roc_auc",
                                roc_auc,
                            )

            for ds_name, ds in metrics.items():
                cur_metrics = None
                result = []
                for method_name, method_data in ds.items():
                    if cur_metrics is None:
                        cur_metrics = list(method_data.keys())
                    row = dict(
                        zip(
                            ["method", *cur_metrics],
                            [method_name]
                            + [
                                f"{np.mean(method_data[metric]):.3f}"
                                for metric in cur_metrics
                            ],
                        )
                    )
                    if method_name == "src_only":
                        result.insert(0, row)
                    else:
                        result.append(row)
                df = pd.DataFrame(result).set_index("method").astype(float)

                if ds_name not in total_dfs.keys():
                    total_dfs[ds_name] = df.sort_index()
                else:
                    total_dfs[ds_name] = pd.concat(
                        [total_dfs[ds_name], df]
                    ).sort_index()

        resulting_df = functools.reduce(
            lambda x, y: pd.concat([x, y], axis=1), total_dfs.values()
        )
        resulting_df.columns = pd.MultiIndex.from_product(
            [[k for k in total_dfs.keys()], list(total_dfs.values())[0].columns],
            names=["data", "metric_type"],
        )

        input_name = resulting_df.columns[0][0]  # type: ignore
        transition = " -> ".join(input_name.split(".")[-2:])  # type: ignore

        final_df = resulting_df.copy()
        final_df.columns = [x[1] for x in final_df.columns]
        final_df = final_df.reset_index()

        # final_df = final_df[~final_df['method'].isin(["mdd", "mcd", "mcc", "dann"])]

        final_df = final_df.melt(id_vars=["method"], var_name="metric")
        final_df["value"] /= 100
        final_df["transition"] = transition
        final_df["antigen"] = (
            "<b>"
            + input_name.split(".")[1]  # type: ignore
            + "</b><br>"
            + " - ".join(input_name.split(".")[2:4])  # type: ignore
        )

        final_df_temp = pd.DataFrame()
        for idx, group_df in final_df.groupby("metric"):
            src_only = group_df[group_df["method"] == "src_only"]["value"].values[0]
            temp_df = group_df.assign(
                symbol=group_df["method"].map(method_to_symbol),
                color=group_df["method"].map(method_to_color),
                text=[
                    f"{src_only:.3g}+{x - src_only:.1g}"
                    if x == group_df["value"].max()
                    else ""
                    for x in group_df["value"]
                ],
            )
            temp_df.loc[temp_df["value"] < src_only, "color"] = "lightgray"
            temp_df["value"] -= src_only
            temp_df = temp_df[temp_df["method"] != "src_only"]

            final_df_temp = pd.concat([final_df_temp, temp_df], ignore_index=True)

        final_df = final_df_temp.copy()
        final_df["metric"] = (
            final_df["metric"]
            .str.replace("accuracy", "Accuracy")
            .str.replace("pr_auc", "PR")
            .str.replace("roc_auc", "ROC")
        )

        return final_df

    def plot(self, final_df: pd.DataFrame, output_dir: Union[Path, str]):
        output_dir = Path(output_dir)

        transition = final_df["transition"].values[0]

        methods = sorted(final_df["method"].unique())
        metrics = sorted(final_df["metric"].unique())
        metrics_order = {metrics[i]: i for i in range(len(metrics))}
        jitter = {
            methods[i]: (i - round(len(methods) / 2)) / 20 for i in range(len(methods))
        }

        n_rows = 1
        n_cols = len(metrics)

        fig = ps.make_subplots(
            rows=n_rows,
            cols=n_cols,
            shared_yaxes="rows",  # type: ignore
            horizontal_spacing=0.075,
            column_titles=[final_df["antigen"].values[0]] * n_cols,
            # subplot_titles=metrics,
        )

        legend_method_ids = []
        for i, metric in enumerate(metrics):
            df = final_df[(final_df["metric"] == metric)]

            for idx, row in df.iterrows():
                showlegend = False
                if (
                    row["color"] != "lightgray"
                    and row["method"] not in legend_method_ids
                ):
                    legend_method_ids.append(row["method"])
                    showlegend = True
                elif (
                    row["color"] == "lightgray"
                    and row["method"] not in legend_method_ids
                    and i == (len(metrics) - 1)
                ):
                    legend_method_ids.append(row["method"])
                    showlegend = True

                sub = go.Scatter(
                    x=[metrics_order[row["metric"]] + jitter[row["method"]]],
                    y=[row["value"]],
                    mode="markers",
                    name=row["method"].upper(),
                    marker=dict(
                        color=row["color"], symbol=row["symbol"], line_width=1, size=15
                    ),
                    showlegend=showlegend,
                    legendrank=methods.index(row["method"]),
                )
                if row["text"]:
                    fig.add_annotation(
                        x=metrics_order[row["metric"]] + jitter[row["method"]],
                        y=row["value"],
                        text=row["text"],
                        showarrow=True,
                        arrowhead=2,
                        ax=20,
                        ay=-20,
                        yshift=10,
                        xshift=5,
                        row=1,
                        col=i + 1,
                    )
                fig.add_trace(sub, row=1, col=i + 1)
            fig.update_xaxes(ticks="", row=1, col=i + 1, title=metric)

        fig.update_xaxes(
            # showticklabels=False,
            tickvals=[0, 1, 2],
            ticktext=[" ", " ", " "],
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        )

        fig.update_yaxes(
            ticks="",
            zeroline=True,
            zerolinecolor="gray",
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            # tickformat="+.2",
            showgrid=True,
        )
        fig.update_yaxes(ticks="outside", row=1, col=1)

        fig.update_annotations(textangle=0, font=dict(size=16), align="center")

        height = 600
        width = 300 * len(metrics)
        fig.update_layout(
            title=f'<span style="font-size:16px;">{transition}</span>',
            title_x=0.5,
            title_y=0.99,
            height=height,
            width=width,
            font=dict(size=16),
            # uniformtext_minsize=16,
            margin=dict(l=25, t=100, b=50, r=25),
            legend=dict(
                # font_size=12,
                orientation="h",
                yanchor="bottom",
                y=-0.4,
                xanchor="center",
                x=0.5,
            ),
            template="simple_white",
        )

        fig.write_image(output_dir / "model_score_comparison.png", scale=2)
        fig.write_html(output_dir / "model_score_comparison.html")
        fig.show(renderer="png")
