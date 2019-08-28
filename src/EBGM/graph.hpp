#pragma once

#include "jet.hpp"

#include <vector>
#include <tuple>
#include <limits>
#include <functional>
#include <cmath>
#include <memory>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#ifndef NDEBUG
#include <sstream>
#include <string>
#include <iomanip>
#endif


// N: The size of jet, same as the 'N' in 'Jet<N>'.
// node: jet
// edge: the distance vector between two jets
template<int N>
class Graph {
    static_assert(N > 0, "The 'N' in Graph<N> must be a positive integer.");

public:
    using Edge = struct {int x; int y;};
    using Node = Jet<N>;

    // When constructed by deserialization:
    //     If not set, the kx and ky of jets will use the value saved in file.
    //     If set, the kx and ky of jets will use these variables directly.
    // When constructed by addNodes():
    //     These variables are ignored and will always be null.
    std::shared_ptr<float[]> m_kx;
    std::shared_ptr<float[]> m_ky;

private:
    // DO NOT modify edges and nodes manually!
    // Use addNode() and replaceNode().
    std::vector<Node> m_nodes;
    std::vector<Edge> m_edges;

    // get the edge between two nodes. 
    // (the edge direction: from smaller node to bigger node)
    Edge &getEdge(int index1, int index2)
    {
        assert(index1 < m_nodes.size() && index1 >= 0);
        assert(index2 < m_nodes.size() && index2 >= 0);
        assert(index1 != index2);

        float index1f, index2f;

        // node1f will always < node2f
        if(index1 > index2) {
            index1f = (float)index2;
            index2f = (float)index1;
        }
        else {
            index1f = (float)index1;
            index2f = (float)index2;
        }


        float edgeIndexf = 0.5F*index2f*(index2f-1.0F) + index1f;
        int edgeIndex = (int)round(edgeIndexf);

        return m_edges[edgeIndex];
    }

    friend class boost::serialization::access;

    template<class Archive>
    void load(Archive & ar, const unsigned int version)
    {
        m_nodes.clear();
        m_edges.clear();

        int nNodes;
        ar & nNodes;

        if(nNodes == 0) {
            return;
        }

        std::shared_ptr<float[]> tmpkx(new float[N]);
        std::shared_ptr<float[]> tmpky(new float[N]);
        float *kx = tmpkx.get();
        float *ky = tmpky.get();
        for(int i=0; i<N; i++) {
            ar & kx[i];
        }
        for(int i=0; i<N; i++) {
            ar & ky[i];
        }

        if(!m_kx || !m_ky){
            m_kx = tmpkx;
            m_ky = tmpky;
        }

        m_nodes.reserve(nNodes);
        for(int i=0; i<nNodes; i++) {
            Jet<N> jet;
            ar & jet.x;
            ar & jet.y;
            ar & jet.a;
            ar & jet.p;
            jet.kx = m_kx;
            jet.ky = m_ky;

            m_nodes.push_back(jet);
        }

        int nEdges;
        ar & nEdges;

        m_edges.reserve(nEdges);
        for(int i=0; i<nEdges; i++) {
            Edge edge;
            ar & edge.x;
            ar & edge.y;
            m_edges.push_back(edge);
        }
    }

    template<class Archive>
    void save(Archive & ar, const unsigned int version) const
    {
        int nNodes = m_nodes.size();
        ar & nNodes;

        if(nNodes == 0) {
            return;
        }

        float *kx = m_nodes[0].kx.get();
        float *ky = m_nodes[0].ky.get();
        for(int i=0; i<N; i++) {
            ar & kx[i];
        }
        for(int i=0; i<N; i++) {
            ar & ky[i];
        }

        for(int i=0; i<nNodes; i++) {
            const Jet<N> &jet = m_nodes[i];
            ar & jet.x;
            ar & jet.y;
            ar & jet.a;
            ar & jet.p;
        }

        int nEdges = m_edges.size();
        ar & nEdges;

        for(int i=0; i<nEdges; i++) {
            Edge edge = m_edges[i];
            ar & edge.x;
            ar & edge.y;
        }
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    constexpr const decltype(m_nodes) &getNodes() const
    {
        return const_cast<const decltype(m_nodes)&>(m_nodes);
    }

    constexpr const decltype(m_edges) &getEdges() const
    {
        return const_cast<const decltype(m_edges)&>(m_edges);
    }

    void clear()
    {
        m_nodes.clear();
        m_edges.clear();
    }

    bool empty() const
    {
        return m_nodes.empty();
    }

    // node will be copied into this struct.
    // This is an O(n) operation.
    void addNode(const Node &node)
    {
        for(auto i: m_nodes) {
            Edge edge;
            edge.x = node.x - i.x;
            edge.y = node.y - i.y;
            m_edges.push_back(edge);
        }
        
        m_nodes.push_back(node);
    }

    // node will be copied into this struct.
    // This is an O(n) operation.
    void replaceNode(const Node &node, int index)
    {
        assert(index < m_nodes.size() && index >=0);

        m_nodes[index] = node;

        int nNodes = m_nodes.size();
        for(int i=0; i<index; i++){
            Edge &edge = getEdge(i, index);
            edge.x = node.x - m_nodes[i].x;
            edge.y = node.y - m_nodes[i].y;
        }
        for(int i=index+1; i<nNodes; i++){
            Edge &edge = getEdge(i, index);
            edge.x = m_nodes[i].x - node.x;
            edge.y = m_nodes[i].y - node.y;
        }
    }

    float compare(const Graph<N> &graph) const
    {
        assert(m_nodes.size() == graph.getNodes().size());

        float sum = 0.0F;
        
        int nNodes = m_nodes.size();
        for(int i=0; i<nNodes; i++){
            sum += m_nodes[i].compare(graph.getNodes()[i]);
        }

        return sum / (float)nNodes;
    }

    //Points<int> toPoints() const
    //{
        //Points<int> ret;
        //for(const auto &i: m_nodes){
            //ret.addPoint({i.x, i.y});
        //}
        //return ret;
    //}





#ifndef NDEBUG
    std::string toString()
    {
        std::ostringstream buff;
        std::string ret;

        char tmp[128];

        buff << "nodes:\n";
        for(int i=0; i<m_nodes.size(); i++){
            sprintf(tmp, "[%d](%d, %d)", 
                i, 
                m_nodes[i].x, 
                m_nodes[i].y
            );
            buff << std::left << std::setw(20) << tmp;
            buff << "\n";
        }

        buff << "\nedges:\n";
        for(int i=0; i<m_edges.size(); i++){
            sprintf(tmp, "[%d](%d, %d)",
                i, 
                m_edges[i].x,
                m_edges[i].y
            );
            buff << std::left << std::setw(20) << tmp;
        }
        buff << "\n";

        ret = buff.str();

        return ret;
    }
    

#endif

};



// N: The size of jet, same as the 'N' in 'Jet<N>'.
// node: consists of the jets on the same fiducial point
// edge: the averaged distance vector
template<int N>
class BunchGraph {
    static_assert(N > 0, "The 'N' in BunchGraph<N> must be a positive integer.");

public:
    using Node = std::vector<typename Graph<N>::Node>;
    using Edge = struct {float x; float y;};

    // Control the scaling of the bunch graph.
    // Has an effect on edge comparing.
    float xScale = 1.0F;
    float yScale = 1.0F;

private:
    // the number of graphs added to this bunch graph.
    int m_nGraphs = 0;

    // DO NOT modify edges and nodes manually!
    // Use addGraph().
    std::vector<Node> m_nodes;
    std::vector<Edge> m_edges;

    // always < 0. 
    float compareEdges(const Graph<N> &graph) const
    {
        assert(m_edges.size() == graph.getEdges().size());

        float sum = 0;
        float nEdges = m_edges.size();

        for(int i=0; i<nEdges; i++) {
            float x1, x2, y1, y2;
            x1 = m_edges[i].x * xScale;
            y1 = m_edges[i].y * yScale;
            x2 = graph.getEdges()[i].x;
            y2 = graph.getEdges()[i].y;

            sum +=
                ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)) /
                (x1*x1 + y1*y1);
            
        }

        return -1.0F * sum / (float)nEdges;
    }

public:
    constexpr const decltype(m_nodes) &getNodes() const
    {
        return const_cast<const decltype(m_nodes)&>(m_nodes);
    }

    constexpr const decltype(m_edges) &getEdges() const
    {
        return const_cast<const decltype(m_edges)&>(m_edges);
    }

    void clear()
    {
        m_nodes.clear();
        m_edges.clear();
        m_nGraphs = 0;
    }

    bool empty() const
    {
        return m_nodes.empty();
    }


    // graph will be copied into this struct.
    void addGraph(const Graph<N> &graph)
    {
        assert(!graph.getNodes().empty());

        // initialize the BunchGraph
        if(m_nGraphs == 0) {
            for(const auto &i: graph.getNodes()){
                Node node;
                node.push_back(i);
                m_nodes.push_back(node);
                node.clear();
            }

            for(const auto &i: graph.getEdges()){
                Edge edge;
                edge.x = i.x;
                edge.y = i.y;
                m_edges.push_back(edge);
            }

            m_nGraphs = 1;

            return;
        }

        assert(m_nodes.size() == graph.getNodes().size());
        assert(m_edges.size() == graph.getEdges().size());

        int nNodes = m_nodes.size();
        int nEdges = m_edges.size();
        for(int i=0; i<nNodes; i++) {
            m_nodes[i].push_back(graph.getNodes()[i]);
        }
        for(int i=0; i<nEdges; i++) {
            m_edges[i].x = 
                (m_nGraphs*m_edges[i].x + graph.getEdges()[i].x) / 
                (m_nGraphs + 1);
            m_edges[i].y = 
                (m_nGraphs*m_edges[i].y + graph.getEdges()[i].y) / 
                (m_nGraphs + 1);
        }

        m_nGraphs++;

    }

    // compare without phase information
    float compare(const Graph<N> &graph) const
    {
        static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
        assert(m_nodes.size() == graph.getNodes().size());
        assert(m_edges.size() == graph.getEdges().size());
        assert(m_nGraphs != 0);

        float sum = 0;

        int nNodes = m_nodes.size();

        for(int i=0; i<nNodes; i++){
            float maxSimi;
            maxSimi = -std::numeric_limits<float>::infinity();

            for(int j=0; j<m_nGraphs; j++){
                float simi;
                simi = m_nodes[i][j].compare(graph.getNodes()[i]);
                if(simi > maxSimi) {
                    maxSimi = simi;
                }
            }

            sum += maxSimi;
        }

        return sum / (float)nNodes;
    }

    float compare(const Graph<N> &graph, float lambda) const
    {
        return compare(graph) + lambda*compareEdges(graph);
    }

    

    // You need to implement a function for estimating displacement with focus:
    // std::tuple<float, float>
    // displacementWithFocus(
    //     const Jet<40> &jet1,
    //     const Jet<40> &jet2,
    //     int focus
    // )
    std::tuple<float/*similarity*/,float/*square sum over displacements*/>
    compareWithPhaseFocus(
        const Graph<N> &graph,
        int focus,
        std::function<
            std::tuple<float,float>(const Jet<N>&,const Jet<N>&,int)
        > dispFunc
    ) const
    {
        static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
        assert(m_nodes.size() == graph.getNodes().size());
        assert(m_edges.size() == graph.getEdges().size());
        assert(m_nGraphs != 0);

        float sumSimi = 0;
        float sumDisp2 = 0;

        int nNodes = m_nodes.size();

        for(int i=0; i<nNodes; i++){
            float maxSimi;
            float minDisp2;
            maxSimi = -std::numeric_limits<float>::infinity();

            for(int j=0; j<m_nGraphs; j++){
                float simi;
                const auto &jet1 = m_nodes[i][j];
                const auto &jet2 = graph.getNodes()[i];
                float dx, dy;

                std::tie(dx, dy) = dispFunc(jet1, jet2, focus);
                
                simi = jet1.compareWithPhase(jet2, dx, dy);

                if(simi > maxSimi) {
                    maxSimi = simi;
                    minDisp2 = dx*dx + dy*dy;
                }
            }

            sumSimi += maxSimi;
            sumDisp2 += minDisp2;
        }

        return std::make_tuple(sumSimi / (float)nNodes, sumDisp2);
    }

    // You need to implement a function for estimating displacement with focus:
    // std::tuple<float, float>
    // displacementWithFocus(
    //     const Jet<40> &jet1,
    //     const Jet<40> &jet2,
    //     int focus
    // )
    std::tuple<float/*similarity*/,float/*square sum over displacements*/>
    compareWithPhaseFocus(
        const Graph<N> &graph,
        int focus,
        std::function<
            std::tuple<float,float> (const Jet<N>&,const Jet<N>&,int)
        > dispFunc,
        float lambda
    ) const
    {
        float simi, sumDisp2;
        std::tie(simi, sumDisp2) = compareWithPhaseFocus(graph, focus, dispFunc);
        simi += lambda*compareEdges(graph);
        return std::make_tuple(simi, sumDisp2); 
    }







#ifndef NDEBUG
    std::string toString()
    {
        std::ostringstream buff;
        std::string ret;

        char tmp[128];

        buff << "nodes:\n";
        for(int i=0; i<m_nodes.size(); i++){
            for(int j=0; j<m_nodes[i].size(); j++){
                sprintf(tmp, "[%d][%d](%d, %d)", 
                    i, 
                    j, 
                    m_nodes[i][j].x, 
                    m_nodes[i][j].y
                );
                buff << std::left << std::setw(20) << tmp;
            }
            buff << "\n";
        }

        buff << "\nedges:\n";
        for(int i=0; i<m_edges.size(); i++){
            sprintf(tmp, "[%d](%.2f, %.2f)",
                i, 
                m_edges[i].x,
                m_edges[i].y
            );
            buff << std::left << std::setw(20) << tmp;
        }
        buff << "\n";

        ret = buff.str();

        return ret;
    }



#endif

};

