xinyan's description
=====
1.  command for audio(fvecs)
./fvecsnndes --input /home/xinyan/programs/data/audio/audio_base.fvecs --output log.txt -K 20 -D 192 -N 53387 --control 1
2.  command for sift1m(fvecs)
./fvecsnndes --input /home/xinyan/programs/data/sift1m/sift1m_base.fvecs --output log.txt -K 20 -D 128 -N 1000000 --control 1 --numthread 10
3.  comand for audio(weidong's format)
./nndes --input audio.data --output log.txt -D 192 --skip 12 --control 1
