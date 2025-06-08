# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import array
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import gzip
import io
import math
import numpy as np
import numpy.typing as npt
from pathlib import Path
import re
import shutil
import tempfile
from typing import NamedTuple
import pysam  # ver 0.22.0
from tqdm import tqdm


# 3-nucleotide context, CG/CHG/CHH etc.

CG_CONTEXT_FORWARD_HASH = {
    'CGA': 'CG',
    'CGT': 'CG',
    'CGC': 'CG',
    'CGG': 'CG',  # 4 CG
    'CAG': 'CHG',
    'CTG': 'CHG',
    'CCG': 'CHG',  # 3 CHG
    'CAA': 'CHH',
    'CAT': 'CHH',
    'CAC': 'CHH',  # 9 CHH
    'CTA': 'CHH',
    'CTT': 'CHH',
    'CTC': 'CHH',
    'CCA': 'CHH',
    'CCT': 'CHH',
    'CCC': 'CHH',
}

CG_CONTEXT_REVERSE_HASH = {
    'ACG': 'CG',
    'TCG': 'CG',
    'CCG': 'CG',
    'GCG': 'CG',  # 4 CG
    'CAG': 'CHG',
    'CTG': 'CHG',
    'CGG': 'CHG',  # 3 CHG
    'AAG': 'CHH',
    'ATG': 'CHH',
    'AGG': 'CHH',  # 9 CHH
    'TAG': 'CHH',
    'TTG': 'CHH',
    'TGG': 'CHH',
    'GAG': 'CHH',
    'GTG': 'CHH',
    'GGG': 'CHH',
}

# 2-nucleotide context of reverse strand

DI_CONTEXT_REVERSE_HASH = {'AG': 'CT', 'TG': 'CA', 'CG': 'CG', 'GG': 'CC'}


def as_bool(x: str):
    x = x.upper()
    if x in ['TRUE', 'T', 'YES', 'Y']:
        return True
    if x in ['FALSE', 'F', 'NO', 'N']:
        return False
    return None


# interval must be involved in single chr
class GenomicInterval(NamedTuple):
    chr: str
    chr_length: int
    start: int
    end: int


# `ref` includes ref bases from `start`-2 to `end`-2
class FaGenomicInterval(NamedTuple):
    chr: str
    start: int
    end: int
    bases: str


class GenomicIntervalGenerator:

    def __init__(
        self,
        fa: pysam.FastaFile,
        chrs: str,
        start: int,
        end: int,
        step: int,
    ) -> None:

        self.chrs = fa.references
        self.lens = fa.lengths
        if chrs == 'all':
            self.chrs_selected = list(self.chrs)
        else:
            self.chrs_selected = chrs.split(',')

        self.start = start
        self.end = end
        self.step = step

        assert step > 0 and start < end
        assert len(self.chrs) > 0 and len(self.chrs) == len(self.lens)

    def __repr__(self):
        return f'GenomicIntervalGenerator({len(self.chrs)} contig(s) with step {self.step})'

    def __iter__(self) -> Generator[GenomicInterval, None, None]:
        for chr, len in zip(self.chrs, self.lens):
            if chr not in self.chrs_selected:
                continue

            end2 = min(self.end, len)
            start = self.start
            end = start + self.step
            while start < end2:
                # if start < end2:
                end = min(end, end2)
                yield GenomicInterval(chr=chr, chr_length=len, start=start, end=end)
                start = end
                end += self.step
                # else:
                #     break


class MyFastaFile(pysam.FastaFile):
    def __init__(self, filename: Path):
        self._fasta = pysam.FastaFile(filename.as_posix())

    @property
    def closed(self) -> bool:
        return self._fasta.closed

    @property
    def filename(self) -> str:
        return self._fasta.filename

    @property
    def references(self) -> Sequence[str]:
        return self._fasta.references

    @property
    def nreferences(self) -> int | None:
        return self._fasta.nreferences

    @property
    def lengths(self) -> Sequence[int]:
        return self._fasta.lengths

    # Delegate methods
    def is_open(self) -> bool:
        return self._fasta.is_open()

    def close(self) -> None:
        self._fasta.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def fetch(
        self,
        reference: str | None = None,
        start: int | None = None,
        end: int | None = None,
        region: str | None = None,
    ) -> str:
        return self._fasta.fetch(reference=reference, start=start, end=end, region=region)

    def get_reference_length(self, reference: str) -> int:
        return self._fasta.get_reference_length(reference)

    def __getitem__(self, reference: str) -> str:
        return self._fasta[reference]

    def __contains__(self, reference: str) -> bool:
        return reference in self._fasta

    def __len__(self) -> int:
        return len(self._fasta)

    def rich_fetch(self, interval: GenomicInterval, padding: int) -> str:
        bases: str
        bases = self.fetch(
            reference=interval.chr,
            start=max(0, interval.start - padding),
            end=min(interval.chr_length, interval.end + padding),
        ).upper()

        # replace non-ATCG letters to N
        bases = re.sub('[^ATCG]', 'N', bases)

        # padding N
        if interval.start < padding:
            bases = 'N' * (padding - interval.start) + bases
        if interval.end + padding > interval.chr_length:
            bases = bases + 'N' * (interval.end + padding - interval.chr_length)

        return bases


class Coverage(NamedTuple):
    watson: npt.NDArray[np.int64]
    crick: npt.NDArray[np.int64]


class Parameters(NamedTuple):
    fa_file: Path
    bam_file: Path
    out_atcg: Path
    out_cg: Path
    out_bed: Path
    tmp_dir: Path
    chr: str
    start: int
    end: int
    step: int
    quality_threshold: int
    context_size: int
    coordinate_base: int  # 0/1-based
    swap_strand: bool  # swap the counts of forward and reverse strands
    read_quality: int
    threads: int


# def reverse_read(read) -> bool:
#     return read.is_reverse

# def forward_read(read) -> bool:
#     return not read.is_reverse


def check_read(forward_read: bool, read_quality: int):
    def valid_read(read: pysam.AlignedSegment):
        return (
            (
                forward_read ^ read.is_reverse
            ) and (
                not read.is_unmapped
            ) and (
                not read.is_duplicate
            ) and (
                not read.is_secondary
            ) and (
                not read.is_qcfail
            ) and (
                read.mapping_quality >= read_quality
            )
        )

    return valid_read


class MyAlignmentFile:
    def __init__(
        self,
        filename: Path,
        mode: str | None = "rb",
        template: pysam.AlignmentFile | None = None,
        reference_names: Sequence[str] | None = None,
        reference_lengths: Sequence[int] | None = None,
        reference_filename: Path | None = None,
        text: str | None = None,
        header: dict | pysam.AlignmentHeader | None = None,
        add_sq_text: bool = False,
        add_sam_header: bool = False,
        check_sq: bool = True,
        index_filename: str | None = None,
        filepath_index: str | None = None,
        require_index: bool = False,
        duplicate_filehandle: bool = False,
        ignore_truncation: bool = False,
        format_options: Sequence[str] | None = None,
        threads: int = 1,
    ):
        """
        Initialize the wrapper class with a pysam.AlignmentFile object.
        """
        self._alignment = pysam.AlignmentFile(
            filename=filename.as_posix(),
            mode=mode,
            template=template,
            reference_names=reference_names,
            reference_lengths=reference_lengths,
            reference_filename=reference_filename.as_posix(),
            text=text,
            header=header,
            add_sq_text=add_sq_text,
            add_sam_header=add_sam_header,
            check_sq=check_sq,
            index_filename=index_filename,
            filepath_index=filepath_index,
            require_index=require_index,
            duplicate_filehandle=duplicate_filehandle,
            ignore_truncation=ignore_truncation,
            format_options=format_options,
            threads=threads,
        )

    # Delegate properties
    @property
    def mapped(self) -> int:
        return self._alignment.mapped

    @property
    def unmapped(self) -> int:
        return self._alignment.unmapped

    @property
    def nocoordinate(self) -> int:
        return self._alignment.nocoordinate

    @property
    def nreferences(self) -> int:
        return self._alignment.nreferences

    @property
    def references(self) -> tuple[str, ...]:
        return self._alignment.references

    @property
    def lengths(self) -> tuple[int, ...]:
        return self._alignment.lengths

    @property
    def reference_filename(self) -> str | None:
        return self._alignment.reference_filename

    @property
    def header(self) -> pysam.AlignmentHeader:
        return self._alignment.header

    # Delegate methods
    def has_index(self) -> bool:
        return self._alignment.has_index()

    def check_index(self) -> bool:
        return self._alignment.check_index()

    def fetch(
        self,
        contig: str | None = None,
        start: int | None = None,
        stop: int | None = None,
        region: str | None = None,
        tid: int | None = None,
        until_eof: bool = False,
        multiple_iterators: bool = False,
        reference: str | None = None,
        end: int | None = None,
    ) -> Iterator:
        return self._alignment.fetch(
            contig=contig,
            start=start,
            stop=stop,
            region=region,
            tid=tid,
            until_eof=until_eof,
            multiple_iterators=multiple_iterators,
            reference=reference,
            end=end,
        )

    def head(self, n: int, multiple_iterators: bool = False) -> Iterator:
        return self._alignment.head(n=n, multiple_iterators=multiple_iterators)

    def mate(self, read: pysam.AlignedSegment) -> pysam.AlignedSegment:
        return self._alignment.mate(read)

    def pileup(
        self,
        contig: str | None = None,
        start: int | None = None,
        stop: int | None = None,
        region: str | None = None,
        reference: str | None = None,
        end: int | None = None,
        truncate: bool = False,
        max_depth: int = 8000,
        stepper: str = "all",
        fastafile: pysam.FastaFile | None = None,
        ignore_overlaps: bool = True,
        flag_filter: int = 1536,
        flag_require: int = 0,
        ignore_orphans: bool = True,
        min_base_quality: int = 13,
        adjust_capq_threshold: int = 50,
        min_mapping_quality: int = 0,
        compute_baq: bool = False,
        redo_baq: bool = False,
    ) -> Iterator:
        return self._alignment.pileup(
            contig=contig,
            start=start,
            stop=stop,
            region=region,
            reference=reference,
            end=end,
            truncate=truncate,
            max_depth=max_depth,
            stepper=stepper,
            fastafile=fastafile,
            ignore_overlaps=ignore_overlaps,
            flag_filter=flag_filter,
            flag_require=flag_require,
            ignore_orphans=ignore_orphans,
            min_base_quality=min_base_quality,
            adjust_capq_threshold=adjust_capq_threshold,
            min_mapping_quality=min_mapping_quality,
            compute_baq=compute_baq,
            redo_baq=redo_baq,
        )

    def count(
        self,
        contig: str | None = None,
        start: int | None = None,
        stop: int | None = None,
        region: str | None = None,
        until_eof: bool = False,
        read_callback: str | Callable[[pysam.AlignedSegment], bool] = "all",
        reference: str | None = None,
        end: int | None = None,
    ) -> int:
        return self._alignment.count(
            contig=contig,
            start=start,
            stop=stop,
            region=region,
            until_eof=until_eof,
            read_callback=read_callback,
            reference=reference,
            end=end,
        )

    def count_coverage(
        self,
        contig: str | None = None,
        start: int | None = None,
        stop: int | None = None,
        region: str | None = None,
        quality_threshold: int = 15,
        read_callback: str | Callable[[pysam.AlignedSegment], bool] = "all",
        reference: str | None = None,
        end: int | None = None,
    ) -> tuple[array.array, array.array, array.array, array.array]:
        return self._alignment.count_coverage(
            contig=contig,
            start=start,
            stop=stop,
            region=region,
            quality_threshold=quality_threshold,
            read_callback=read_callback,
            reference=reference,
            end=end,
        )

    def find_introns_slow(
        self, read_iterator: Iterable[pysam.AlignedSegment]
    ) -> dict[tuple[int, int], int]:
        return self._alignment.find_introns_slow(read_iterator)

    def find_introns(
        self, read_iterator: Iterable[pysam.AlignedSegment]
    ) -> dict[tuple[int, int], int]:
        return self._alignment.find_introns(read_iterator)

    def close(self) -> None:
        self._alignment.close()

    def write(self, read: pysam.AlignedSegment) -> int:
        return self._alignment.write(read)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __iter__(self) -> Iterator:
        return iter(self._alignment)

    def __next__(self) -> pysam.AlignedSegment:
        return next(self._alignment)

    def is_valid_tid(self, tid: int) -> bool:
        return self._alignment.is_valid_tid(tid)

    def get_tid(self, reference: str) -> int:
        return self._alignment.get_tid(reference)

    def get_reference_name(self, tid: int) -> str:
        return self._alignment.get_reference_name(tid)

    def get_reference_length(self, reference: str) -> int:
        return self._alignment.get_reference_length(reference)

    def get_index_statistics(self) -> list:
        return self._alignment.get_index_statistics()

    def Watson_Crick_coverage(
        self, interval: GenomicInterval, params: Parameters
    ) -> Coverage:
        cov_watson = self.count_coverage(
            contig=interval.chr,
            start=interval.start,
            stop=interval.end,
            quality_threshold=params.quality_threshold,
            read_callback=check_read(
                forward_read=True, read_quality=params.read_quality
            ),
        )
        cov_crick = self.count_coverage(
            contig=interval.chr,
            start=interval.start,
            stop=interval.end,
            quality_threshold=params.quality_threshold,
            read_callback=check_read(
                forward_read=False, read_quality=params.read_quality
            ),
        )

        # in some bams, the read strandness seem be reversly flaged in `FLAG`
        # parsed in read.is_reverse
        # for example gemBS

        if params.swap_strand:
            return Coverage(crick=np.array(cov_watson), watson=np.array(cov_crick))
        else:
            return Coverage(watson=np.array(cov_watson), crick=np.array(cov_crick))


def myOpenFile(file: Path):
    if file.name.endswith('.gz'):
        outfile = gzip.open(file, 'wt')
    else:
        outfile = io.open(file, 'wt')
    return outfile


def process_interval(interval: GenomicInterval,
                     fa_file: Path,
                     bam_file: Path,
                     params: Parameters) -> dict[str, str] | None:
    with (MyFastaFile(fa_file) as fa,
          MyAlignmentFile(filename=bam_file, reference_filename=fa_file, mode='rb') as bam,
          tempfile.NamedTemporaryFile(mode='w+', suffix='.actg',
                                      dir=params.tmp_dir, delete=False) as tmp_atcg,
          tempfile.NamedTemporaryFile(mode='w+', suffix='.cg',
                                      dir=params.tmp_dir, delete=False) as tmp_cg,
          tempfile.NamedTemporaryFile(mode='w+', suffix='.bed',
                                      dir=params.tmp_dir, delete=False) as tmp_bed):
        tmp_paths: dict[str, str] = {
            'atcg': tmp_atcg.name, 'cg': tmp_cg.name, 'bed': tmp_bed.name}
        # context size
        con_size = params.context_size
        # ref sequences
        bases = fa.rich_fetch(interval, padding=con_size - 1).upper()
        # bam coverages
        try:
            coverage = bam.Watson_Crick_coverage(interval, params)
            cov_sum_W = np.sum(coverage.watson, axis=0)
            cov_sum_C = np.sum(coverage.crick, axis=0)
        except KeyError:
            return

        # chr, base, start, type, dinucleotide, beta,
        # depth, m_count, A_waston, T_watson, C_watson, G_watson,
        # A_crick, T_crick, C_crick, G_crick)

        for i in range(interval.end - interval.start):
            if cov_sum_W[i] + cov_sum_C[i] == 0:
                continue
            j = i + 2
            if j >= len(bases):
                continue
            base = bases[j]
            if base not in {'C', 'G'}:
                continue
            nCT = coverage.watson[1, i] + coverage.watson[3, i]
            nGA = coverage.crick[2, i] + coverage.crick[0, i]

            if (base == 'C' and nCT == 0) or (base == 'G' and nGA == 0):
                continue

            chr = interval.chr
            base = base
            pos = interval.start + i + params.coordinate_base

            A_watson = coverage.watson[0, i]
            T_watson = coverage.watson[3, i]
            C_watson = coverage.watson[1, i]
            G_watson = coverage.watson[2, i]
            A_crick = coverage.crick[0, i]
            T_crick = coverage.crick[3, i]
            C_crick = coverage.crick[1, i]
            G_crick = coverage.crick[2, i]
            if base == 'C':
                # CG/CHG/CHH
                bases_con = bases[j: (j + con_size)]
                type = '--' if 'N' in bases_con else CG_CONTEXT_FORWARD_HASH[bases_con]
                dinucleotide = bases[j: (j + 2)]
                beta = coverage.watson[1, i] / nCT
                depth = coverage.watson[1, i]
                m_count = nCT

            else:
                bases_con = bases[(j - con_size + 1): (j + 1)]
                bases2 = bases[(j - 1): (j + 1)]
                type = '--' if 'N' in bases_con else CG_CONTEXT_REVERSE_HASH[bases_con]
                dinucleotide = '--' if 'N' in bases2 else DI_CONTEXT_REVERSE_HASH[bases2]
                beta = coverage.crick[2, i] / nGA
                depth = coverage.crick[2, i]
                m_count = nGA

            tmp_cg.write(f'{chr}\t{base}\t{pos}\t{type}\t{dinucleotide}\t{beta}\t{m_count}\t{depth}\n')
            if type == 'CG':
                tmp_bed.write(f'{chr}\t{pos}\t{pos + 1}\t{beta * 100}\n')
            tmp_atcg.write(
                f'{chr}\t{base}\t{pos}\t{type}\t{dinucleotide}\t'
                f'{beta}\t{m_count}\t{depth}\t'
                f'{A_watson}\t{A_crick}\t{T_watson}\t{T_crick}\t'
                f'{C_watson}\t{C_crick}\t{G_watson}\t{G_crick}\n')
    return tmp_paths


def methylextractor(params: Parameters) -> None:
    Path(params.tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(params.out_atcg).parent.mkdir(parents=True, exist_ok=True)
    Path(params.out_cg).parent.mkdir(parents=True, exist_ok=True)
    Path(params.out_bed).parent.mkdir(parents=True, exist_ok=True)

    outfile_atcg = myOpenFile(params.out_atcg)
    outfile_cg = myOpenFile(params.out_cg)
    outfile_bed = myOpenFile(params.out_bed)

    outfile_cg.write('chr\tbase\tpos\ttype\tdinucleotide\tbeta\tm_count\tdepth\n')
    outfile_bed.write('chr\tstart\tend\tbeta\n')
    outfile_atcg.write('chr\tbase\tpos\ttype\tdinucleotide\t'
                       'beta\tm_count\tdepth\t'
                       'A_watson\tA_crick\tT_watson\tT_crick\t'
                       'C_watson\tC_crick\tG_watson\tG_crick\n')

    intervals = GenomicIntervalGenerator(
        MyFastaFile(params.fa_file),
        chrs=params.chr,
        start=params.start,
        end=params.end,
        step=params.step
    )

    with ProcessPoolExecutor(max_workers=params.threads) as executor:
        futures: list[Future[dict[str, str]]] = [
            executor.submit(process_interval, interval,
                            params.fa_file, params.bam_file, params)
            for interval in intervals]

        tmp_path_lists: list[dict[str, str]] = []
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc='Processing intervals'):
            tmp_path_lists.append(future.result())

    for tmp_files in tqdm(tmp_path_lists,
                          total=len(tmp_path_lists),
                          desc='Merging tmp files'):
        with open(tmp_files['atcg'], 'r') as f:
            shutil.copyfileobj(f, outfile_atcg)
        with open(tmp_files['cg'], 'r') as f:
            shutil.copyfileobj(f, outfile_cg)
        with open(tmp_files['bed'], 'r') as f:
            shutil.copyfileobj(f, outfile_bed)

        for f in tmp_files.values():
            Path(f).unlink(missing_ok=True)

    outfile_atcg.close()
    outfile_cg.close()
    outfile_bed.close()


if __name__ == '__main__':
    # parse command line
    usage = 'Usage: methylExtrator -b sample.bam -g genome.fa [options]'
    desc = 'Extract ATCG (ATCGmap) and CG (CGmap/bedgraph) profiles from bam file'

    parser = ArgumentParser(description=desc)
    parser.add_argument('-b', '--bam-file', dest='in_bam',
                        type=str, required=True, help='an input .bam file')
    parser.add_argument('-g', '--reference-genome', dest='in_fa',
                        type=str, required=True,
                        help='genome reference .fa file with index (.fai) in the same path',)
    parser.add_argument('--output-atcgmap', dest='out_atcg', type=str,
                        required=True, help='output of ATCGmap file')
    parser.add_argument('--output-cgmap', dest='out_cg', type=str, required=True,
                        help='output of CGmap file')
    parser.add_argument('--output-bed', dest='out_bed', type=str, default='',
                        required=True, help='output of bedgraph file')
    parser.add_argument('--tmp-dir', dest='tmp_dir', type=str, default='',
                        help='temporary directory for intermediate files, default is current directory')
    parser.add_argument('--chr', dest='chr', type=str, default='all',
                        help='chromosomes/contigs')
    parser.add_argument('--start', dest='start', type=int, default=0,
                        help='start coordinate of chromosomes/contigs')
    parser.add_argument('--end', dest='end', type=int, default=math.inf,
                        help='end coordinate of chromosomes/contigs')
    parser.add_argument('--batch-size', dest='step', type=int, default=2000000,
                        help='batch size of genomic intervals')
    parser.add_argument('--swap-strand', dest='swap_strand', action='store_true',
                        help='swap read counts on two strands, true/false, or yes/no')
    parser.add_argument('--base-quality', dest='base_quality', type=int, default=0,
                        help='base sequencing quality threshold')
    parser.add_argument('--read-quality', dest='read_quality', type=int, default=0,
                        help='read mapping quality threshold')
    parser.add_argument('--coordinate-base', dest='coordinate_base', type=int, default=1,
                        help='0/1-based coordinate of output')
    parser.add_argument('--threads', dest='threads', type=int, default=1,
                        help='number of threads to use')

    options = parser.parse_args()

    # check params
    assert (
        sum(x == '-' for x in [options.out_atcg, options.out_cg, options.out_bed]) <= 1
    ), 'can only set one output to stdout at most'

    params = Parameters(
        fa_file=Path(options.in_fa),
        bam_file=Path(options.in_bam),
        out_atcg=Path(options.out_atcg),  # three types of outputs
        out_cg=Path(options.out_cg),
        out_bed=Path(options.out_bed),
        tmp_dir=Path(options.tmp_dir),
        chr=options.chr,  # 'all' for all chrs
        start=options.start,
        end=options.end,
        # end=50_000_000,
        step=options.step,
        quality_threshold=options.base_quality,  # base seq quality
        read_quality=options.read_quality,  # read mapping quality threshold
        context_size=3,  # size of CHG/...
        coordinate_base=options.coordinate_base,  # 0/1-based
        swap_strand=options.swap_strand,  # swap counts of two strands
        threads=options.threads
    )
    methylextractor(params)
