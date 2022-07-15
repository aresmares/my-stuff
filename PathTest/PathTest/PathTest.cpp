// PathTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

struct Point
{
    int x;
    int y;
    int value = 0;

    bool Is(Point a)
    {
        if (a.x == x && a.y == y)
            return true;
        else
            return false;
    }
    bool Isin(std::vector<Point> points)
    {
        for (Point i : points)
        {
            if (this->Is(i))
                return true;
        }
        return false;
    }

    double Distance(Point b)
    {
        return sqrt(pow(b.x - this->x, 2) + pow(b.y - this->y, 2));

    }

    Point GetClosestPoint(std::vector<Point> points) {

        Point closestPoint;
        double minDist = 1000.00;
        for (Point i : points)
        {
            double dist = this->Distance(i);
            if (dist < minDist)
            {
                minDist = dist;
                closestPoint = i;
            }
        }
        return closestPoint;
    }
};

class Grid
{
    public:

        int size;

        std::vector<Point> grid;

        Grid()
        {

        }

        Grid(int gridSize) 
        {

            size = gridSize;
        
            for (int x = 0; x < size; x++)
            {
                for (int y = 0; y < size; y++)
                {
                    grid.push_back({x,y,0});
                }
            }

        }

        void SetValue (int x, int y, int value) {
            grid[x*size +y].value = value;
        }
        void SetValue(Point a, int value) {
            grid[a.x * size + a.y].value = value;
        }

        int GetValue(int x, int y) {
            return grid[x * size + y].value;
        }

        int GetValue(Point a) {
            return grid[a.x * size + a.y].value;
        }

        void drawGrid()
        {
            for (int x = 0; x < size; x++)
            {
                for (int y = 0; y < size; y++)
                {
                    std::cout << GetValue(x,y) << " ";
                }
                std::cout << " \n";

            }
        }

        void WriteToFile(std::string filename) {
            std::ofstream myfile;
            myfile.open(filename);

            for (int x = 0; x < this->size; x++)
            {
                for (int y = 0; y < this->size; y++)
                {
                    myfile << this->GetValue(x, y) << " ";
                }
                myfile << " \n";
            }
            myfile.close();

        }

        void ApplyPath(std::vector<Point> path, int value=1) {
            for (Point i : path) {
                this->SetValue(i, value);
            }

        }
        std::vector<Point> getNeighbours(Point a, bool diagonal=true, bool back=true) {
            std::vector<Point> neightbours;

            // horizontal + vertical

            if (back == true) {
                if (a.y - 1 >= 0)
                    neightbours.push_back(Point{ a.x, a.y - 1 });
                if (a.x - 1 >= 0)
                    neightbours.push_back(Point{ a.x - 1, a.y });
            }
            if (a.x + 1 < size)
                neightbours.push_back(Point{ a.x + 1, a.y });
            if (a.y + 1 < size)
                neightbours.push_back(Point{ a.x, a.y + 1});
            
            // diagonals
            if (diagonal == true) {
                if (back == true) {
                    if (a.x - 1 >= 0 && a.y - 1 >= 0)
                        neightbours.push_back(Point{ a.x - 1, a.y - 1 });
                }
                if (a.x - 1 >= 0 && a.y + 1 < size)
                    neightbours.push_back(Point{ a.x - 1, a.y + 1 });

                if (a.x + 1 < size && a.y - 1 < size)
                    neightbours.push_back(Point{ a.x + 1, a.y - 1 });

                if (a.x + 1 < size && a.y + 1 < size)
                    neightbours.push_back(Point{ a.x + 1, a.y + 1 });
            } 
            return neightbours;
        }
};


class PathMaker
{
public:
    Grid grid;
    Point startPoint;
    Point endPoint;
    double threshold = 10;

    std::vector<Point> generatedPath;
    std::vector<Point> directPath;

    int endVal = 5;
    int startVal = 5;
    int pathVal = 9;



    PathMaker(Point start, Point end, Grid map, double threshold)
    {
        this->startPoint = start;
        this->endPoint = end;
        this->grid = map;
        this->threshold = threshold;
        
        this->directPath = GetDirectPath();
        this->generatedPath = GetRandomPath();

        grid.SetValue(start, startVal);
        grid.SetValue(end, endVal);

    }


    std::vector<Point> GetRandomPath()
    {
        std::vector<Point> path;
        path.push_back(startPoint);
        std::vector<Point> directPath = this->directPath;

        Point currentPoint = startPoint;

        while (currentPoint.Distance(endPoint) > 1)
        {
            std::vector<Point> neighbours = grid.getNeighbours(currentPoint, false, true);

            for (Point i : neighbours) {
                int nextIdx = rand() % neighbours.size();
                Point nextNode = neighbours[nextIdx];
                Point closestDirectPoint = nextNode.GetClosestPoint(directPath);
                double distance = closestDirectPoint.Distance(nextNode);
                if ((distance < this->threshold) && !(nextNode.Isin(path))) {
                    currentPoint = nextNode;
 //                   std::cout << "ClosestPoint: (" << closestDirectPoint.x << "," << closestDirectPoint.y << ") ";
                    break;
                }
            }
            if (!currentPoint.Isin(neighbours)) {
                currentPoint = endPoint.GetClosestPoint(path);
            }

            path.push_back(currentPoint);
            grid.SetValue(currentPoint, pathVal);

//            std::cout << "At: (" << currentPoint.x << "," << currentPoint.y << ") \n";
            
        }
        this->generatedPath = path;
        grid.drawGrid();

        return path;

    }

    std::vector<Point> GetDirectPath()
    {
        std::vector<Point> path;
        Point currentPoint = startPoint;

        while (currentPoint.Distance(endPoint) > 1)
        {
            std::vector<Point> neighbours = grid.getNeighbours(currentPoint);
            Point nextPoint = currentPoint;
            double minDist = grid.size*10.0;
            for (Point point : neighbours)
            {
                double dist = point.Distance(endPoint);
                if (dist < minDist)
                {
                    minDist = dist;
                    nextPoint = point;
                }
            }
            currentPoint = nextPoint;
            path.push_back(currentPoint);
        }
        return path;
    }
};


int main()
{
    srand(time(NULL));

    int SIZE = 50;
    double THRESHOLD = 50;

    Grid test(SIZE);
    Point start{0, SIZE/4 -1 };
    Point end{ SIZE-1, SIZE/2 -1 };

    PathMaker pt(start, end, test, THRESHOLD);
    pt.grid.ApplyPath(pt.generatedPath);
    //pt.grid.ApplyPath(pt.directPath);

    pt.grid.WriteToFile("example.txt");

    return 0;
}
