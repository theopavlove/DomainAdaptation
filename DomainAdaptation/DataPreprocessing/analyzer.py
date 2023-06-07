from pathlib import Path
from typing import Union

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as ps

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


title_filters = [
    "<b>TBX21</b>: Blood - Th1_Cells",
]


class Analyzer:
    def generate_total_scores(self, path_to_results: Union[Path, str]) -> pd.DataFrame:
        total_score_df = pd.DataFrame()
        path_to_results = Path(path_to_results)

        for directory in sorted(path_to_results.iterdir()):
            dirname = directory.name
            antigen_type = dirname.split(".", maxsplit=1)[0]
            if antigen_type == "TFs":
                results_tsv = directory / f"{dirname}.tsv"

                tf = dirname.split(".")[-3]
                cell_type_class = dirname.split(".")[-2]
                cell_type = dirname.split(".")[-1]

                score_df = (
                    pd.read_table(results_tsv, header=[0, 1])
                    .drop([0])
                    .set_index(("data", "metric_type"))
                    .rename_axis(index=None)
                    .sort_index()
                )

                score_df.columns = pd.MultiIndex.from_tuples(
                    [
                        (
                            tf,
                            cell_type_class,
                            cell_type,
                            f"<b>{tf}</b>: {cell_type_class} - {cell_type}".strip(
                                " - "
                            ),
                            ".".join(tup[0].split(".")[-2:]),
                            tup[1],
                        )
                        for tup in score_df.columns
                    ]
                )
                total_score_df = pd.concat([total_score_df, score_df], axis=1)

        total_score_df = total_score_df.melt(
            var_name=[
                "tf",
                "cell_class",
                "cell_type",
                "full_title",
                "transition",
                "metric",
            ],  # type: ignore
            ignore_index=False,
        ).reset_index(names="method")

        total_score_df = total_score_df[
            total_score_df["transition"].str.contains("mm10.hg38|hg38.mm10")
        ]
        total_score_df["value"] /= 100
        return total_score_df

    def plot_scatters(
        self,
        metric,
        titles,
        transitions,
        new_plotly_df,
        jitter,
        full_titles_order,
        methods,
        full_titles,
    ):
        n_rows = 2
        n_cols = 6
        fig = ps.make_subplots(
            rows=n_rows,
            cols=n_cols,
            shared_yaxes="rows",
            horizontal_spacing=0.01,
            vertical_spacing=0.03,
            column_titles=[x.replace(":", "<br>") for x in titles["full_title"].values],
            row_titles=transitions,
        )

        legend_method_ids = []
        for i, transition in enumerate(transitions):
            for j, full_title in enumerate(full_titles):
                df = new_plotly_df[
                    (new_plotly_df["transition"] == transition)
                    & (new_plotly_df["full_title"] == full_title)
                ]

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
                        and i == (len(transitions) - 1)
                        and j == (len(full_titles) - 1)
                    ):
                        legend_method_ids.append(row["method"])
                        showlegend = True
                    sub = go.Scatter(
                        x=[
                            full_titles_order[row["full_title"]] + jitter[row["method"]]
                        ],
                        y=[row["value"]],
                        mode="markers",
                        name=row["method"].upper(),
                        marker=dict(
                            color=row["color"],
                            symbol=row["symbol"],
                            line_width=1,
                            size=15,
                        ),
                        showlegend=showlegend,
                        legendrank=methods.index(row["method"]),
                    )
                    if row["text"]:
                        fig.add_annotation(
                            x=full_titles_order[row["full_title"]]
                            + jitter[row["method"]],
                            y=row["value"],
                            text=row["text"],
                            showarrow=True,
                            arrowhead=2,
                            ax=20,
                            ay=-20,
                            yshift=10,
                            xshift=5,
                            row=i + 1,
                            col=j + 1,
                        )
                    fig.add_trace(sub, row=i + 1, col=j + 1)
                fig.update_xaxes(ticks="", row=i + 1, col=j + 1)

        fig.update_xaxes(
            showticklabels=False,
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
        fig.update_yaxes(ticks="outside", row=2, col=1)

        fig.update_annotations(textangle=0, font=dict(size=16), align="center")
        fig.for_each_annotation(
            lambda a: a.update(y=1.01)
            if a.text in [x.replace(":", "<br>") for x in titles["full_title"].values]
            else a.update(x=-0.13)
            if a.text in transitions
            else ()
        )

        height = 400 * len(transitions)
        width = 300 * len(titles)
        # metric_str = metric.replace("accuracy", "Accuracy").replace("pr_auc", "PR AUC").replace("roc_auc", "ROC AUC")
        fig.update_layout(
            # title = f"<b>{metric_str}</b> score gains in <b>transcription factor</b> predictions for each domain adaptation method in comparison with the source-only trained model",
            # title_x=0,
            # title_y=.98,
            height=height,
            width=width,
            font=dict(size=16),
            # uniformtext_minsize=16,
            margin=dict(l=225, t=75, b=50, r=25),
            legend=dict(
                # font_size=12,
                orientation="h",
                yanchor="bottom",
                y=-0.075,
                xanchor="right",
                x=0.98,
            ),
            template="simple_white",
        )
        fig.write_image(f"./img/six_tfs_scatter_{metric}.png", scale=4)

        print(f"{height}x{width}")
        fig.show(renderer="png")
