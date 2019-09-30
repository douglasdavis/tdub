TARGET := augment-npy

all: $(TARGET)

cnpy.o: cnpy.cpp
	g++ -pthread -m64 -std=c++17 -O3 -fPIC -c $<

augment-npy.o: augment-npy.cpp
	 g++ -pthread -m64 -std=c++17 -O3 -fPIC -I$(shell root-config --incdir) -c $<

$(TARGET): augment-npy.o cnpy.o
	 g++ -o $@ $^ -L$(shell root-config --libdir) -lCore -lRIO -lTree -lImt -lm -ldl -rdynamic -lz

.PHONY: clean

clean:
	$(RM) *.o $(TARGET)
