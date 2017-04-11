test:
	python -m pytest -s

memory:
	mprof run --multiprocess -T 0.01 tests/test_pairwise_distance.py
	mprof plot

save_plot:
	mprof plot -o memory_profile.png

clean:
	rm mprofile_*.dat