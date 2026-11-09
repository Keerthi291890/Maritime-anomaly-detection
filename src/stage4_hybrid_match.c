
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#define MAX_REGIONS 4
#define EARTH_RADIUS_KM 6371.0088
#define PI 3.14159265358979323846
#define SEARCH_RADIUS_KM 25.0

const char *regions[MAX_REGIONS] = {
    "argentina",
    "ecuador_galapagos",
    "ghana",
    "senegal"
};

#pragma pack(push, 1)
typedef struct {
    uint64_t record_id;
    int32_t day;
    int64_t vessel_code;
    double lat;
    double lon;
} Record;
#pragma pack(pop)

_Static_assert(sizeof(Record) == 36,
               "Binary Record must be exactly 36 bytes");

typedef struct {
    uint64_t sar_id;
    uint64_t ais_id;
    double distance_km;
    int method;
    int status;
} Result;

typedef struct {
    Result *items;
    size_t count;
    size_t capacity;
} ResultVector;


static void append_result(ResultVector *v, Result result) {
    if (v->count == v->capacity) {
        size_t new_capacity = v->capacity == 0 ? 256 : v->capacity * 2;

        Result *new_items = realloc(
            v->items, new_capacity * sizeof(Result)
        );

        if (!new_items) {
            fprintf(stderr, "ERROR: Result allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        v->items = new_items;
        v->capacity = new_capacity;
    }

    v->items[v->count++] = result;
}


static int compare_records(const void *a, const void *b) {
    const Record *x = a;
    const Record *y = b;

    if (x->day < y->day) return -1;
    if (x->day > y->day) return 1;

    if (x->vessel_code < y->vessel_code) return -1;
    if (x->vessel_code > y->vessel_code) return 1;

    return 0;
}


static double to_radians(double degrees) {
    return degrees * PI / 180.0;
}


static double haversine(const Record *a, const Record *b) {
    double lat1 = to_radians(a->lat);
    double lat2 = to_radians(b->lat);

    double dlat = lat2 - lat1;
    double dlon = to_radians(b->lon - a->lon);

    double sin_lat = sin(dlat / 2.0);
    double sin_lon = sin(dlon / 2.0);

    double h = sin_lat * sin_lat
             + cos(lat1) * cos(lat2) * sin_lon * sin_lon;

    if (h > 1.0) h = 1.0;
    if (h < 0.0) h = 0.0;

    return 2.0 * EARTH_RADIUS_KM * asin(sqrt(h));
}


static Record *read_records(
    const char *path,
    size_t *count
) {
    FILE *file = fopen(path, "rb");

    if (!file) {
        perror(path);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "ERROR: Cannot seek %s\n", path);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    long bytes = ftell(file);

    if (bytes < 0 || bytes % (long)sizeof(Record) != 0) {
        fprintf(stderr, "ERROR: Invalid binary file size: %s\n", path);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    rewind(file);

    *count = (size_t)bytes / sizeof(Record);

    Record *records = malloc((*count) * sizeof(Record));

    if (!records && *count > 0) {
        fprintf(stderr, "ERROR: Input allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    if (fread(records, sizeof(Record), *count, file) != *count) {
        fprintf(stderr, "ERROR: Failed reading %s\n", path);
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    fclose(file);

    return records;
}


/* First AIS record whose day is >= target day. */
static size_t lower_bound_day(
    const Record *records,
    size_t count,
    int32_t day
) {
    size_t left = 0;
    size_t right = count;

    while (left < right) {
        size_t mid = left + (right - left) / 2;

        if (records[mid].day < day)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}


/* First AIS record whose (day, vessel_code) is >= target pair. */
static size_t lower_bound_pair(
    const Record *records,
    size_t count,
    int32_t day,
    int64_t vessel_code
) {
    size_t left = 0;
    size_t right = count;

    while (left < right) {
        size_t mid = left + (right - left) / 2;

        int before =
            records[mid].day < day ||
            (records[mid].day == day &&
             records[mid].vessel_code < vessel_code);

        if (before)
            left = mid + 1;
        else
            right = mid;
    }

    return left;
}


static void match_one_sar(
    const Record *sar,
    const Record *ais,
    size_t ais_count,
    ResultVector *results
) {
    /* Case 1: SAR has a usable vessel ID. */
    if (sar->vessel_code != 0) {

        size_t start = lower_bound_pair(
            ais,
            ais_count,
            sar->day,
            sar->vessel_code
        );

        double best_distance = INFINITY;
        uint64_t best_id = 0;
        int found = 0;

        for (size_t i = start;
             i < ais_count &&
             ais[i].day == sar->day &&
             ais[i].vessel_code == sar->vessel_code;
             i++) {

            double distance = haversine(sar, &ais[i]);

            if (distance < best_distance) {
                best_distance = distance;
                best_id = ais[i].record_id;
                found = 1;
            }
        }

        if (found) {
            Result r = {
                sar->record_id,
                best_id,
                best_distance,
                1,  /* same_vessel_id */
                1   /* matched */
            };

            append_result(results, r);
        } else {
            Result r = {
                sar->record_id,
                0,
                -1.0,
                1,
                0
            };

            append_result(results, r);
        }

        return;
    }

    /* Case 2: Missing SAR ID. Search all same-day AIS records. */
    size_t start = lower_bound_day(
        ais, ais_count, sar->day
    );

    size_t candidate_count = 0;

    for (size_t i = start;
         i < ais_count && ais[i].day == sar->day;
         i++) {

        double distance = haversine(sar, &ais[i]);

        if (distance <= SEARCH_RADIUS_KM) {
            Result r = {
                sar->record_id,
                ais[i].record_id,
                distance,
                2,  /* spatiotemporal */
                1
            };

            append_result(results, r);
            candidate_count++;
        }
    }

    if (candidate_count == 0) {
        Result r = {
            sar->record_id,
            0,
            -1.0,
            2,
            0
        };

        append_result(results, r);
    }
}


static void write_results(
    const char *output_dir,
    int rank,
    int thread_id,
    const char *region,
    const ResultVector *results
) {
    char path[1024];

    snprintf(
        path,
        sizeof(path),
        "%s/matches_%s_rank%d_thread%d.csv",
        output_dir,
        region,
        rank,
        thread_id
    );

    FILE *file = fopen(path, "w");

    if (!file) {
        perror(path);
        MPI_Abort(MPI_COMM_WORLD, 4);
    }

    fprintf(
        file,
        "sar_record_id,ais_record_id,distance_km,"
        "match_method,match_status\n"
    );

    for (size_t i = 0; i < results->count; i++) {
        const Result *r = &results->items[i];

        fprintf(
            file,
            "%llu,%llu,%.8f,%s,%s\n",
            (unsigned long long)r->sar_id,
            (unsigned long long)r->ais_id,
            r->distance_km,
            r->method == 1 ? "same_vessel_id" : "spatiotemporal",
            r->status == 1 ? "matched" : "unmatched"
        );
    }

    fclose(file);
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != MAX_REGIONS) {
        if (rank == 0) {
            fprintf(
                stderr,
                "ERROR: Run with exactly 4 MPI ranks, one per region.\n"
            );
        }

        MPI_Finalize();
        return 1;
    }

    if (argc != 3) {
        if (rank == 0) {
            fprintf(
                stderr,
                "Usage: mpirun -np 4 ./stage4_hybrid_match "
                "<input_dir> <output_dir>\n"
            );
        }

        MPI_Finalize();
        return 1;
    }

    const char *input_dir = argv[1];
    const char *output_dir = argv[2];
    const char *region = regions[rank];

    char ais_path[1024];
    char sar_path[1024];

    snprintf(
        ais_path, sizeof(ais_path),
        "%s/ais_presence/%s.bin",
        input_dir, region
    );

    snprintf(
        sar_path, sizeof(sar_path),
        "%s/sar_presence/%s.bin",
        input_dir, region
    );

    size_t ais_count = 0;
    size_t sar_count = 0;

    Record *ais = read_records(ais_path, &ais_count);
    Record *sar = read_records(sar_path, &sar_count);

    qsort(
        ais,
        ais_count,
        sizeof(Record),
        compare_records
    );

    int thread_count = omp_get_max_threads();

    ResultVector *vectors = calloc(
        (size_t)thread_count,
        sizeof(ResultVector)
    );

    if (!vectors) {
        fprintf(stderr, "ERROR: Thread result allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 5);
    }

    double start_time = MPI_Wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic, 16)
        for (long long i = 0; i < (long long)sar_count; i++) {
            match_one_sar(
                &sar[i],
                ais,
                ais_count,
                &vectors[tid]
            );
        }
    }

    double elapsed = MPI_Wtime() - start_time;

    size_t total_results = 0;

    for (int t = 0; t < thread_count; t++) {
        write_results(
            output_dir,
            rank,
            t,
            region,
            &vectors[t]
        );

        total_results += vectors[t].count;
        free(vectors[t].items);
    }

    free(vectors);
    free(ais);
    free(sar);

    printf(
        "Rank %d | Region: %s | AIS: %zu | SAR: %zu | "
        "Results: %zu | Threads: %d | Time: %.4f s\n",
        rank,
        region,
        ais_count,
        sar_count,
        total_results,
        thread_count,
        elapsed
    );

    MPI_Finalize();

    return 0;
}
