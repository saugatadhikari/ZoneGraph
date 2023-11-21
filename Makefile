CC=g++
CFLAGS=-std=c++14 -O3 -march=native -lgdal -I /home/saugat/.conda/envs/gdal_venv/include -L /home/saugat/.conda/envs/gdal_venv/lib -ltbb -fopenmp -lpthread
OBJDIR=objs/

INC=GeotiffRead.h GeotiffWrite.h DataTypes.h
	
all: zone_graph

HMF_parallel: $(OBJDIR)zone_graph.o $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ 

$(OBJDIR)zone_graph.o: zone_graph.cpp $(INC)
	$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf zone_graph $(OBJDIR)

MKDIR_P = mkdir -p

