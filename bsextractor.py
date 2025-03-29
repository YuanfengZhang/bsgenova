#!/usr/bin/env python
from argparse import ArgumentParser
import array
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import gzip
import io
import math
import numpy as np
import numpy.typing as npt
import re
import sys
from typing import Callable, Dict, Generator, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union
import pysam  # ver 0.22.0


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
        chrs,
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
    def __init__(self, filename: str):
        self._fasta = pysam.FastaFile(filename)

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
    def nreferences(self) -> Optional[int]:
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
        reference: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        region: Optional[str] = None,
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
    watson: npt.NDArray
    crick: npt.NDArray


class Parameters(NamedTuple):
    fa_file: str
    bam_file: str
    out_atcg: str
    out_cg: str
    out_bed: str
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
        filename: str,
        mode: Optional[str] = "rb",
        template: Optional[pysam.AlignmentFile] = None,
        reference_names: Optional[Sequence[str]] = None,
        reference_lengths: Optional[Sequence[int]] = None,
        reference_filename: Optional[str] = None,
        text: Optional[str] = None,
        header: Optional[Union[Dict, pysam.AlignmentHeader]] = None,
        add_sq_text: bool = False,
        add_sam_header: bool = False,
        check_sq: bool = True,
        index_filename: Optional[str] = None,
        filepath_index: Optional[str] = None,
        require_index: bool = False,
        duplicate_filehandle: bool = False,
        ignore_truncation: bool = False,
        format_options: Optional[Sequence[str]] = None,
        threads: int = 1,
    ):
        """
        Initialize the wrapper class with a pysam.AlignmentFile object.
        """
        self._alignment = pysam.AlignmentFile(
            filename=filename,
            mode=mode,
            template=template,
            reference_names=reference_names,
            reference_lengths=reference_lengths,
            reference_filename=reference_filename,
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
    def references(self) -> Tuple[str, ...]:
        return self._alignment.references

    @property
    def lengths(self) -> Tuple[int, ...]:
        return self._alignment.lengths

    @property
    def reference_filename(self) -> Optional[str]:
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
        contig: Optional[str] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        region: Optional[str] = None,
        tid: Optional[int] = None,
        until_eof: bool = False,
        multiple_iterators: bool = False,
        reference: Optional[str] = None,
        end: Optional[int] = None,
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
        contig: Optional[str] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        region: Optional[str] = None,
        reference: Optional[str] = None,
        end: Optional[int] = None,
        truncate: bool = False,
        max_depth: int = 8000,
        stepper: str = "all",
        fastafile: Optional[pysam.FastaFile] = None,
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
        contig: Optional[str] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        region: Optional[str] = None,
        until_eof: bool = False,
        read_callback: Union[str, Callable[[pysam.AlignedSegment], bool]] = "all",
        reference: Optional[str] = None,
        end: Optional[int] = None,
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
        contig: Optional[str] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        region: Optional[str] = None,
        quality_threshold: int = 15,
        read_callback: Union[str, Callable[[pysam.AlignedSegment], bool]] = "all",
        reference: Optional[str] = None,
        end: Optional[int] = None,
    ) -> Tuple[array.array, array.array, array.array, array.array]:
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
    ) -> Dict[Tuple[int, int], int]:
        return self._alignment.find_introns_slow(read_iterator)

    def find_introns(
        self, read_iterator: Iterable[pysam.AlignedSegment]
    ) -> Dict[Tuple[int, int], int]:
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

    def get_index_statistics(self) -> List:
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


def myOpenFile(file: str):
    if file == '':
        return None
    if file == '-':
        outfile = sys.stdout
    elif file.endswith('.gz'):
        outfile = gzip.open(file, 'wt')
    else:
        outfile = io.open(file, 'wt')
    return outfile


def process_interval(interval: GenomicInterval,
                     fa_file: str,
                     bam_file: str,
                     params: Parameters) -> Optional[List[Tuple[str, str, int,
                                                                str, str, float,
                                                                int, int, int,
                                                                int, int, int,
                                                                int, int, int,
                                                                int]]]:
    with MyFastaFile(fa_file) as fa, MyAlignmentFile(bam_file, 'rb') as bam:
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
            return None

        results = []
        # (chr,
        #  base,
        #  start,
        #  gc_context,
        #  dinucleotide,
        #  beta,
        #  depth,
        #  m_count,
        #  A_waston,
        #  T_watson,
        #  C_watson,
        #  G_watson,
        #  A_crick,
        #  T_crick,
        #  C_crick,
        #  G_crick)

        for i in range(interval.end - interval.start):
            if cov_sum_W[i] + cov_sum_C[i] == 0:
                continue

            j = i + 2
            base = bases[j]
            if base == 'C':
                nCT = coverage.watson[1, i] + coverage.watson[3, i]
                if nCT > 0:
                    # CG/CHG/CHH
                    bases_con = bases[j: (j + con_size)]
                    results.append((interval.chr,
                                    base,
                                    interval.start + i + params.coordinate_base,
                                    '--' if 'N' in bases_con else CG_CONTEXT_FORWARD_HASH[bases_con],
                                    bases[j: (j + 2)],
                                    coverage.watson[1, i] / nCT,
                                    coverage.watson[1, i],
                                    nCT,
                                    coverage.watson[0, i], coverage.watson[3, i],
                                    coverage.watson[1, i], coverage.watson[2, i],
                                    coverage.crick[0, i], coverage.crick[3, i],
                                    coverage.crick[1, i], coverage.crick[2, i]))

            elif base == 'G':
                nGA = coverage.crick[2, i] + coverage.crick[0, i]
                if nGA > 0:
                    bases_con = bases[(j - con_size + 1): (j + 1)]
                    CG_context = ('--' if 'N' in bases_con else CG_CONTEXT_REVERSE_HASH[bases_con])
                    bases2 = bases[(j - 1): (j + 1)]
                    results.append((interval.chr,
                                    base,
                                    interval.start + i + params.coordinate_base,
                                    CG_context,
                                    '--' if 'N' in bases2 else DI_CONTEXT_REVERSE_HASH[bases2],
                                    coverage.crick[2, i] / nGA,
                                    coverage.crick[2, i],
                                    nGA,
                                    coverage.watson[0, i], coverage.watson[3, i],
                                    coverage.watson[1, i], coverage.watson[2, i],
                                    coverage.crick[0, i], coverage.crick[3, i],
                                    coverage.crick[1, i], coverage.crick[2, i]))
            else:
                pass

        return results


def methylExtractor(params: Parameters) -> None:
    outfile_atcg = myOpenFile(params.out_atcg)
    outfile_cg = myOpenFile(params.out_cg)
    outfile_bed = myOpenFile(params.out_bed)

    outfile_cg.write('chr\tbase\tpos\tgc_context\tdinucleotide\tbeta\tm_count\tdepth\n')
    outfile_bed.write('chr\tstart\tend\tbeta\n')
    outfile_atcg.write('chr\tbase\tpos\tgc_context\tdinucleotide\t'
                       'beta\tm_count\tdepth\t'
                       'A_watson\tA_crick\tT_watson\tT_crick\t'
                       'C_watson\tC_crick\tG_watson\tG_crick\n')

    intervals = list(
        GenomicIntervalGenerator(
            MyFastaFile(params.fa_file),
            chrs=params.chr,
            start=params.start,
            end=params.end,
            step=params.step
        )
    )

    with ProcessPoolExecutor(max_workers=params.threads) as executor:
        futures: List[Future] = [executor.submit(process_interval,
                                                 interval,
                                                 params.fa_file,
                                                 params.bam_file,
                                                 params)
                                 for interval in intervals]
        for future in as_completed(futures):
            if future.result():
                results: List[Tuple[str, str, int,
                                    str, str, float,
                                    int, int, int,
                                    int, int, int,
                                    int, int, int,
                                    int]] = [i for i in future.result() if i]
                for result in [i for i in results if i]:
                    (chr, base, pos,
                     gc_context, dinucleotide,
                     beta, depth, m_count,
                     A_watson, T_watson,
                     C_watson, G_watson,
                     A_crick, T_crick,
                     C_crick, G_crick) = result
                    if outfile_cg and depth > 0:
                        outfile_cg.write(
                            f'{chr}\t{base}\t{pos}\t{gc_context}\t{dinucleotide}\t{beta}\t{m_count}\t{depth}\n')
                    if outfile_bed and depth > 0 and gc_context == 'CG':
                        outfile_bed.write(f'{chr}\t{pos}\t{pos + 1}\t{beta * 100}\n')
                    if outfile_atcg:
                        outfile_atcg.write(
                            f'{chr}\t{base}\t{pos}\t{gc_context}\t{dinucleotide}\t'
                            f'{beta}\t{m_count}\t{depth}\t'
                            f'{A_watson}\t{A_crick}\t{T_watson}\t{T_crick}\t'
                            f'{C_watson}\t{C_crick}\t{G_watson}\t{G_crick}\n')

    if (outfile_atcg is not None) and (outfile_atcg != '-'):
        outfile_atcg.close()
    if (outfile_cg is not None) and (outfile_cg != '-'):
        outfile_cg.close()
    if (outfile_bed is not None) and (outfile_bed != '-'):
        outfile_bed.close()


if __name__ == '__main__':
    # parse command line
    usage = 'Usage: methylExtrator -b sample.bam -g genome.fa [options]'
    desc = 'Extract ATCG (ATCGmap) and CG (CGmap/bedgraph) profiles from bam file'

    parser = ArgumentParser(description=desc)
    parser.add_argument(
        '-b',
        '--bam-file',
        dest='in_bam',
        help='an input .bam file',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-g',
        '--reference-genome',
        dest='in_fa',
        help='genome reference .fa file with index (.fai) in the same path',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output-atcgmap',
        dest='out_atcg',
        help='output of ATCGmap file',
        type=str,
        required=False,
        default='',
    )
    parser.add_argument(
        '--output-cgmap',
        dest='out_cg',
        help='output of CGmap file',
        type=str,
        required=False,
        default='',
    )
    parser.add_argument(
        '--output-bed',
        dest='out_bed',
        help='output of bedgraph file',
        type=str,
        required=False,
        default='',
    )
    parser.add_argument(
        '--chr', dest='chr', help='chromosomes/contigs', type=str, default='all'
    )
    parser.add_argument(
        '--start',
        dest='start',
        help='start coordinate of chromosomes/contigs',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--end',
        dest='end',
        help='end coordinate of chromosomes/contigs',
        type=int,
        default=math.inf,
    )
    parser.add_argument(
        '--batch-size',
        dest='step',
        help='batch size of genomic intervals',
        type=int,
        default=2_000_000,
    )

    parser.add_argument(
        '--swap-strand',
        dest='swap_strand',
        help='swap read counts on two strands, true/false, or yes/no',
        type=as_bool,
        required=False,
        default='no',
    )
    parser.add_argument(
        '--base-quality',
        dest='base_quality',
        help='base sequencing quality threshold',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--read-quality',
        dest='read_quality',
        help='read mapping quality threshold',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--coordinate-base',
        dest='coordinate_base',
        help='0/1-based coordinate of output',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--threads',
        dest='threads',
        help='parallel',
        type=int,
        default=1,
    )

    options = parser.parse_args()

    # check params
    assert (
        sum(x == '-' for x in [options.out_atcg, options.out_cg, options.out_bed]) <= 1
    ), 'can only set one output to stdout at most'

    params = Parameters(
        fa_file=options.in_fa,
        bam_file=options.in_bam,
        out_atcg=options.out_atcg,  # three types of outputs
        out_cg=options.out_cg,
        out_bed=options.out_bed,
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
    methylExtractor(params)
