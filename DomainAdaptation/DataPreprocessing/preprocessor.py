# import logging
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Union

import pandas as pd

from .utils.cmdrun import cmdrun


class Preprocessor:
    def check_lengths(self, path: str):
        return (
            pd.read_table(path, header=None)
            .assign(length=lambda x: x[2] - x[1])["length"]
            .nunique()
        )

    def test_preprocessed_files(self, original: str, randomized: str):
        if not os.path.exists(f"{original}"):
            print(f"No data for: {original}")
            return

        same_length_original = self.check_lengths(f"{original}")
        assert (
            same_length_original == 1
        ), "Original file contains different region lengths!"

        same_length_randomized = self.check_lengths(f"{randomized}")
        assert (
            same_length_randomized == 1
        ), "Randomized file contains different region lengths!"

        sp = cmdrun(
            f"bedtools intersect -a {original} -b {randomized} -u | wc -l", shell=True
        )
        intersection_size = int(sp.stdout.decode("UTF-8"))
        assert intersection_size == 0, "Original file overlaps the randomized!"

    def get_fasta_from_bed(
        self,
        bed_input: Union[Path, str],
        dir_genome: Union[Path, str],
        assembly: str,
        bedtools_bin: str = "bedtools",
    ):
        fasta_output = str(bed_input).replace(".bed", ".fa")

        with tempfile.TemporaryDirectory() as tmpdir:
            path_to_genome = Path(tmpdir) / "genome.fa"

            # ungzip a path_source_genome to tmp_genome
            with open(path_to_genome, "wb") as f_out:
                cmdrun(f"gunzip -c {dir_genome}/{assembly}.fa.gz", stdout=f_out)

            with open(fasta_output, "w") as f_out:
                cmdrun(
                    (
                        f"{bedtools_bin} getfasta "
                        f"-fi {str(path_to_genome)} "
                        f"-bed {str(bed_input)}"
                    ),
                    stdout=f_out,
                )

    def make_uniform_lengths(self, s: pd.Series, length: int):
        difference = (s[2] - s[1] - length) / 2
        if difference > 0:
            return pd.Series(
                [s[0], s[1] + math.floor(difference), s[2] - math.ceil(difference)],
                index=s.index,
                name=s.name,
            )
        return pd.Series(pd.NA, index=s.index, name=s.name)

    def run_bedtools_shuffle(
        self,
        bed_input: str,
        exclude: str,
        chrom_sizes: str,
        f_out,
        bedtools_bin: str = "bedtools",
    ):
        return cmdrun(
            cmd=(
                f"{bedtools_bin} shuffle "
                f"-i {bed_input} -excl {exclude} -g {chrom_sizes} "
                f"-seed 42 -maxTries 1000000 -noOverlapping"
            ),
            stdout=f_out,
        )

    def generate_random_samples(
        self,
        dir_input: Union[Path, str],
        dir_genome: Union[Path, str],
        get_fasta: bool = True,
        bedtools_bin: str = "bedtools",
    ):
        for root_entry in sorted(os.scandir(dir_input), key=lambda x: x.name):
            if str(root_entry.name)[0] == ".":
                continue
            for bed_input in sorted(os.listdir(root_entry.path)):
                if (".fa" in bed_input) or ("random" in bed_input):
                    continue

                bed_input_path = f"{root_entry.path}/{bed_input}"
                bed_output_path = bed_input_path.replace(".bed", ".random.bed")
                if os.path.exists(bed_output_path):
                    logging.warning(f"Already exists, so skipping: {bed_input}")
                    continue

                print(f"Generating random samples for {bed_input}")

                assembly = bed_input.rsplit(".", maxsplit=2)[-2]
                bed_output_2x_path = (
                    f"{root_entry.path}/{bed_input.replace('.bed', '.random_2x.bed')}"
                )

                with tempfile.TemporaryDirectory() as tmpdir:
                    bed_exclude_path = f"{tmpdir}/excl.bed"
                    bed_input_2x_path = f"{tmpdir}/input2x.bed"

                    with open(bed_exclude_path, "w") as f_out:
                        cmdrun(
                            f"cat {bed_input_path}",
                            stdout=f_out,
                        )
                    with open(bed_input_2x_path, "w") as f_out:
                        cmdrun(f"cat {bed_input_path} {bed_input_path}", stdout=f_out)

                    with open(bed_output_path, "w") as f_out:
                        self.run_bedtools_shuffle(
                            bed_input_path,
                            bed_exclude_path,
                            f"{dir_genome}/{assembly}.chrom.sizes",
                            f_out,
                            bedtools_bin,
                        )
                    with open(bed_output_2x_path, "w") as f_out:
                        self.run_bedtools_shuffle(
                            bed_input_2x_path,
                            bed_exclude_path,
                            f"{dir_genome}/{assembly}.chrom.sizes",
                            f_out,
                            bedtools_bin,
                        )

                    self.test_preprocessed_files(bed_input_path, bed_output_path)
                    self.test_preprocessed_files(bed_input_path, bed_output_2x_path)

                    if get_fasta:
                        self.get_fasta_from_bed(
                            bed_input=bed_output_path,
                            dir_genome=dir_genome,
                            assembly=assembly,
                        )
                        self.get_fasta_from_bed(
                            bed_input=bed_output_2x_path,
                            dir_genome=dir_genome,
                            assembly=assembly,
                        )

    def preprocess_bed(
        self,
        dir_input: Union[Path, str],
        dir_output: Union[Path, str],
        dir_genome: Union[Path, str],
        length: int = 1000,
        get_fasta: bool = True,
        bedtools_bin: str = "bedtools",
    ):
        dir_input = Path(dir_input)
        dir_output = Path(dir_output)
        dir_genome = Path(dir_genome)

        for root_entry in sorted(os.scandir(dir_input), key=lambda x: x.name):
            preprocessed_dir = (
                dir_output / f"{root_entry.name.rsplit('.', maxsplit=1)[0]}"
            )
            os.makedirs(preprocessed_dir, exist_ok=True)

            bed_output = preprocessed_dir / f"{root_entry.name}.bed"
            if os.path.exists(bed_output):
                logging.warning(
                    f"Already exists, so skipping: {os.path.basename(bed_output)}"
                )
                continue

            print(f"Preprocessing {root_entry.name:36}")

            experiment_list = [
                f"{root_entry.path}/{x}" for x in os.listdir(root_entry.path)
            ]
            sp1 = cmdrun(f"cat {' '.join(experiment_list)}")
            sp2 = cmdrun("cut -f1,2,3", input=sp1.stdout)
            sp3 = cmdrun("sort -k1,1 -k2,2n", input=sp2.stdout)
            sp4 = cmdrun(f"{bedtools_bin} merge -i stdin", input=sp3.stdout)

            assembly = root_entry.name.rsplit(".", maxsplit=1)[1]
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_bed_input = f"{tmpdir}/tmp.bed"
                with open(tmp_bed_input, "w") as f_out:
                    cmd = (
                        f"{bedtools_bin} slop "
                        f"-i stdin "
                        f"-g {dir_genome}/{assembly}.chrom.sizes "
                        f"-b {str(length // 2)}"
                    )
                    cmdrun(cmd, input=sp4.stdout, stdout=f_out)

                output_df = (
                    pd.read_table(tmp_bed_input, header=None)
                    .apply(lambda x: self.make_uniform_lengths(x, length), axis=1)
                    .dropna()
                )
                output_df.to_csv(bed_output, sep="\t", header=False, index=False)

                size_input = int(
                    cmdrun(f"wc -l {tmp_bed_input}", shell=True)
                    .stdout.decode("utf-8")
                    .split()[0]
                )
                size_output = int(
                    cmdrun(f"wc -l {bed_output}", shell=True)
                    .stdout.decode("utf-8")
                    .split()[0]
                )
                print(
                    f"#{size_input:>7,d} -> #{size_output:>7,d} (-{(size_input-size_output)/size_input:>4.2%})."  # noqa
                )

                if get_fasta:
                    self.get_fasta_from_bed(
                        bed_input=bed_output, dir_genome=dir_genome, assembly=assembly
                    )
