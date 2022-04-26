python read_childes_by_age.py > ../../data/results/corpus_stats/no_hall_corpus_stats.txt &
python read_childes_by_decade.py &
python read_childes_by_class.py > ../../data/results/corpus_stats/hall_corpus_stats.txt &
python read_childes_parentchild.py &
python compute_total_corpus_stats.py > ../../data/results/corpus_stats/full_corpus_stats.txt &
python read_switchboard.py &
