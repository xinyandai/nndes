/* 
    Copyright (C) 2010,2011 Wei Dong <wdong.pku@gmail.com>. All Rights Reserved.

    DISTRIBUTION OF THIS PROGRAM IN EITHER BINARY OR SOURCE CODE FORM MUST BE
    PERMITTED BY THE AUTHOR.
*/

#include <sys/time.h>
#include <iomanip>
#include <boost/tr1/random.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <nndes.h>
#include <nndes-data.h>

using namespace std;
using namespace boost;
using namespace nndes;

namespace po = boost::program_options; 

class Timer
{
    struct  timeval start; 
public:
    Timer () {}
    /// Start timing.
    void tick ()
    {
        gettimeofday(&start, 0); 
    }
    /// Stop timing, return the time passed (in second).
    float tuck (const char *msg) const
    {
        struct timeval end;
        float   diff; 
        gettimeofday(&end, 0); 

        diff = (end.tv_sec - start.tv_sec) 
                + (end.tv_usec - start.tv_usec) * 0.000001; 
        if (msg != 0) {
            std::cout << msg << ':' <<  diff << "s\n";
        }
        return diff;
    }
};

int main (int argc, char *argv[]) {
    string input_path;
    string output_path;

    int num_thread = 1;

    int D;
    int N;
    int K;
    int I;
    float T;
    float S;
    float noise;

    int control;

    int skip;
    int gap;

    bool lshkit = false;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("input", po::value(&input_path), "input path")
    ("output", po::value(&output_path), "output path")
    (",K", po::value(&K)->default_value(20), "number of nearest neighbor")
    ("fast", "use fast configuration (less accurate)")
    ("control", po::value(&control)->default_value(0), "number of control points")
    ("numthread", po::value(&num_thread)->default_value(8), "number of openMP thread")
    ("dim,D", po::value(&D), "dimension, see format")
    ("number,N", po::value(&N), "number, number fo vector")
    ("skip", po::value(&skip)->default_value(0), "see format")
    ("gap", po::value(&gap)->default_value(0), "see format")
    ("lshkit", "use LSHKIT data format");

    po::options_description desc_hidden("Expert options");
    desc_hidden.add_options()
    ("iteration,I", po::value(&I)->default_value(100), "expert")
    ("rho,S", po::value(&S)->default_value(1.0), "expert")
    ("delta,T", po::value(&T)->default_value(0.001), "expert")
    ("noise", po::value(&noise)->default_value(0), "expert")
    ;

    po::options_description desc("Allowed options");
    desc.add(desc_visible).add(desc_hidden);

    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("lshkit") == 1) {
        lshkit = true;
    }

    if (vm.count("fast") && (vm.count("rho") == 0)) {
        S = 0.5;
    }

    if (vm.count("help")) {
        cout << "Usage: nndes {-D DIMENSION | --lshkit} [OTHER OPTIONS]... INPUT [OUTPUT]" << endl;
        cout << "Construct k-nearest neighbor graph for Euclidean spaces using L2 distance as similarity measure..\n" << endl;
        cout << desc_visible << endl;
        cout << "Input Format:" << endl;
        cout << "  The INPUT file is parsed as a architecture-dependent binary file.  The initial <skip> bytes are skipped.  After that, every <D * sizeof(float)> bytes are read as a D-dimensional vector.  There could be an optional <gap>-byte gap between each vectors.  Therefore, the program expect the file to contain [size(INPUT)-skip]/[D*sizeof(float)+gap] vectors.\n"
                "  If the option \"--lshkit\" is specified, the initial 3*sizeof(int) bytes are interpreted as three 32-bit integers: sizeof(float), number of vectors in the file and the dimension.  The program then sets D = dimension, skip = 3 * sizeof(int) and gap = 0.\n"  << endl;
        cout << "Output Format:" << endl;
        cout << "  Each input vector is assigned an serial ID (0, 1, ...) according to the order they appear in the input.  Each output line contains the ID of a point followed by the K IDs of its nearest neighbor.\n" << endl;
        cout << "Control:" << endl;
        cout << "  To measure the accuracy of the algorithm, <control> points are randomly sampled, and their k-nearest neighbors are found with brute-force search.  The control is then used to measure the recall of the main algorithm.\n" << endl;
        cout << "Progress Report:" << endl;
        cout << "  The following parameters are reported after each iteration:\n"
                "  update: update rate of the K * N result entries.\n"
                "  recall: estimated recall, or 0 if no control is specified.\n"
                "  cost: number of similarity evaluate / [N*(N-1)/2, the brute force cost].\n";
        return 0;
    }

    if (vm.count("input") == 0) {
        cerr << "Missing input. Run \"nndes -h\" to see usage information." << endl;
        return 1;
    }

    if (vm.count("dim") == 0 && !lshkit) {
        cerr << "Missing dimension." << endl;
        return 1;
    }

    FloatDataset data(D, N);
    data.loadFvecs(input_path);
    assert( N == data.size());

    omp_set_num_threads(num_thread);

    OracleL2<FloatDataset> oracle(data);

    if (noise != 0) {
        tr1::ranlux64_base_01 rng;
        float sum = 0, sum2 = 0;
        for (int i = 0; i < data.size(); ++i) {
            for (int j = 0; j < data.getDim(); ++j) {
                sum += data[i][j];
                sum2 += sqr(data[i][j]);
            }
        }
        float total = float(data.size()) * data.getDim();
        float avg2 = sum2 / total, avg = sum / total;
        float dev = sqrt(avg2 - avg * avg);
        cerr << "Adding Gaussian noise w/ " << noise << "x sigma(" << dev << ")..." << endl;
        boost::normal_distribution<float> gaussian(0, noise * dev);
        for (int i = 0; i < data.size(); ++i) {
            for (int j = 0; j < data.getDim(); ++j) {
                data[i][j] += gaussian(rng);
            }
        }
    }
    

    // Generate control points
    // control points are randomly sampled points whose K-NN are found by brute-force search
    // these points are used to evaluate the accuracy of the nn-descent algorithm.
    vector<int> control_index;
    Dataset<int, sizeof(int)> control_knn;

    if (control > 0) {
        cerr << "Generating control points..." << endl;
        Timer prep_timer;
        prep_timer.tick();
        // random sample control points
        control_index.resize(data.size());
        {
            int i = 0;
            BOOST_FOREACH(int &v, control_index) {
                v = i++;
            }
            random_shuffle(control_index.begin(), control_index.end());
            control_index.resize(control);
        }

        vector<KNN> knns(control);
        BOOST_FOREACH(KNN &knn, knns) {
            knn.init(K);
        }

        progress_display progress(data.size(), cerr);
#pragma omp parallel for
        for (int i = 0; i < data.size(); ++i) {
            for (int j = 0; j < control; ++j) {
                if (i == control_index[j]) continue;
                knns[j].update(KNN::Element(i, oracle(control_index[j], i)));
            }
#pragma omp critical 
            ++progress;
        }

        control_knn.reset(K, control);
        for (int i = 0; i < control; ++i) {
            int *p = control_knn[i];
            for (int j = 0; j < K; ++j) {
                p[j] = knns[i][j].key;
            }
        }
        prep_timer.tuck("preprocess timer");
    }

    cerr << "Starting NN-Descent..." << endl;

    Timer timer;
    timer.tick();


    NNDescent<OracleL2<FloatDataset> > nndes(N, K, S, oracle, GRAPH_BOTH);

    float total = float(N) * (N - 1) / 2;
    cout.precision(5);
    cout.setf(ios::fixed);
    for (int it = 0; it < I; ++it) {

	Timer iter_timer;
	iter_timer.tick();

    	int t = nndes.iterate();
	    float iter_time = iter_timer.tuck(0);
        float rate = float(t) / (K * data.size());

        float recall = 0;
        if (control) {  // report accuracy
            const vector<KNN> &nn = nndes.getNN();
#pragma omp parallel for default(shared) reduction(+:recall) 
            for (int i = 0; i < control_knn.size(); ++i) {
                recall += nndes::recall(control_knn[i], nn[control_index[i]], K);
            }
            recall /= control;
        }
        cout << setw(2) << it << " update:" << rate << " recall:" << recall << " cost:" << float(nndes.getCost())/total  << endl;
        std::cout << "iteration time " << ':' <<  iter_time << " seconds\n";
	if (rate < T) break;
    }

    cout << boost::format("Construction time: %1%s.") % timer.tuck(0) << endl;

    if (vm.count("output")) {
        const vector<KNN> &nn = nndes.getNN();
        ofstream os(output_path.c_str());
        int i = 0;
        BOOST_FOREACH(KNN const &knn, nn) {
            os << i++;
            BOOST_FOREACH(KNN::Element const &e, knn) {
                os << ' ' << e.key;
            }
            os << endl;
        }
        os.close();
    }

    return 0;
}

