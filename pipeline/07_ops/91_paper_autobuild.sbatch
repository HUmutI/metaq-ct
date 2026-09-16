#!/bin/bash
#SBATCH --job-name=paperBuild
#SBATCH --partition=bch-compute,bch-compute-pe
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/temp_work/ch278233/ts_logs/paperBuild_%j.out
# The evaluation jobs rewrite paper/generated/*.tex on their own schedule, and
# the PDF silently goes stale until someone recompiles. This polls the fragments
# and rebuilds any paper whose inputs moved. It only ever writes PDFs, never
# .tex, so it cannot fight an editor.
set -uo pipefail
P=/home/ch278233/paper
export XDG_CACHE_HOME=/temp_work/ch278233/PEDS_BENCH/_tex/cache
T=/temp_work/ch278233/PEDS_BENCH/_tex/tectonic
INTERVAL=${PAPER_BUILD_INTERVAL:-180}

newest_input() {   # newest mtime among a paper's own source and every fragment
  { stat -c %Y "$P/$1.tex" 2>/dev/null; stat -c %Y "$P"/generated/*.tex 2>/dev/null; } | sort -rn | head -1
}

echo "[autobuild] start $(date), polling every ${INTERVAL}s"
while true; do
  for doc in pediatric main adult; do
    [ -f "$P/$doc.tex" ] || continue
    src=$(newest_input "$doc")
    pdf=$(stat -c %Y "$P/$doc.pdf" 2>/dev/null || echo 0)
    if [ "${src:-0}" -gt "$pdf" ]; then
      echo "[autobuild] $(date +%H:%M:%S) $doc.pdf stale, rebuilding"
      if $T -X compile "$P/$doc.tex" --outdir "$P" >/tmp/autobuild_$doc.log 2>&1; then
        pages=$(pdfinfo "$P/$doc.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
        unres=$(pdftotext -layout "$P/$doc.pdf" - 2>/dev/null | grep -c '??')
        echo "[autobuild]   ok: $doc.pdf $pages pages, $unres unresolved refs"
      else
        echo "[autobuild]   FAILED: $doc, see /tmp/autobuild_$doc.log"
        grep -iE "^! |^error" /tmp/autobuild_$doc.log | head -3
      fi
    fi
  done
  sleep "$INTERVAL"
done
