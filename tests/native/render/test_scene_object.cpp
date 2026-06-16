// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/scene_object.h"
#include "falcor2/render/geometry.h"
#include "falcor2/render/geometry/static_mesh_geometry.h"
#include "falcor2/render/geometry/geometry_group.h"

using namespace falcor;

TEST_SUITE_BEGIN("SceneObjectCollectionView");

TEST_CASE("empty iteration")
{
    SceneObjectCollection<Geometry> collection(nullptr);
    SceneObjectCollectionView<Geometry> view(collection);

    CHECK(view.begin() == view.end());

    size_t count = 0;
    for (auto* obj : view)
        ++count;
    CHECK_EQ(count, 0);
}

TEST_CASE("iterate objects")
{
    SceneObjectCollection<Geometry> collection(nullptr);

    auto g0 = ref<Geometry>(new Geometry());
    auto g1 = ref<Geometry>(new Geometry());
    auto g2 = ref<Geometry>(new Geometry());
    collection.add(g0);
    collection.add(g1);
    collection.add(g2);

    SceneObjectCollectionView<Geometry> view(collection);

    // Check size matches.
    CHECK_EQ(view.size(), 3);
    CHECK(view.begin() != view.end());

    // Collect pointers via range-based for loop.
    std::vector<Geometry*> iterated;
    for (auto* obj : view)
        iterated.push_back(obj);

    REQUIRE_EQ(iterated.size(), 3);
    CHECK_EQ(iterated[0], g0.get());
    CHECK_EQ(iterated[1], g1.get());
    CHECK_EQ(iterated[2], g2.get());
}

TEST_CASE("const iteration")
{
    SceneObjectCollection<Geometry> collection(nullptr);

    auto g0 = ref<Geometry>(new Geometry());
    auto g1 = ref<Geometry>(new Geometry());
    collection.add(g0);
    collection.add(g1);

    const SceneObjectCollectionView<Geometry> view(collection);

    std::vector<const Geometry*> iterated;
    for (const auto* obj : view)
        iterated.push_back(obj);

    REQUIRE_EQ(iterated.size(), 2);
    CHECK_EQ(iterated[0], g0.get());
    CHECK_EQ(iterated[1], g1.get());
}

TEST_CASE("manual iterator usage")
{
    SceneObjectCollection<Geometry> collection(nullptr);

    auto g0 = ref<Geometry>(new Geometry());
    auto g1 = ref<Geometry>(new Geometry());
    collection.add(g0);
    collection.add(g1);

    SceneObjectCollectionView<Geometry> view(collection);

    auto it = view.begin();
    CHECK_EQ(*it, g0.get());
    ++it;
    CHECK_EQ(*it, g1.get());
    ++it;
    CHECK(it == view.end());
}

TEST_CASE("find by name")
{
    SceneObjectCollection<Geometry> collection(nullptr);

    auto g0 = ref<Geometry>(new Geometry());
    g0->set_name("alpha");
    auto g1 = ref<Geometry>(new Geometry());
    g1->set_name("beta");
    auto g2 = ref<Geometry>(new Geometry());
    g2->set_name("alpha");
    collection.add(g0);
    collection.add(g1);
    collection.add(g2);

    SceneObjectCollectionView<Geometry> view(collection);

    // Returns first match.
    CHECK_EQ(view.find("alpha"), g0.get());
    CHECK_EQ(view.find("beta"), g1.get());
    // Returns nullptr for no match.
    CHECK_EQ(view.find("gamma"), nullptr);
}

TEST_CASE("find by type")
{
    SceneObjectCollection<Geometry> collection(nullptr);

    auto mesh = ref<StaticMeshGeometry>(new StaticMeshGeometry());
    auto group = ref<GeometryGroup>(new GeometryGroup());
    auto mesh2 = ref<StaticMeshGeometry>(new StaticMeshGeometry());
    collection.add(mesh);
    collection.add(group);
    collection.add(mesh2);

    SceneObjectCollectionView<Geometry> view(collection);

    // Find first by type.
    CHECK_EQ(view.find<StaticMeshGeometry>(), mesh.get());
    CHECK_EQ(view.find<GeometryGroup>(), group.get());
}

TEST_CASE("find by type and name")
{
    SceneObjectCollection<Geometry> collection(nullptr);

    auto mesh = ref<StaticMeshGeometry>(new StaticMeshGeometry());
    mesh->set_name("obj");
    auto group = ref<GeometryGroup>(new GeometryGroup());
    group->set_name("obj");
    auto mesh2 = ref<StaticMeshGeometry>(new StaticMeshGeometry());
    mesh2->set_name("obj2");
    collection.add(mesh);
    collection.add(group);
    collection.add(mesh2);

    SceneObjectCollectionView<Geometry> view(collection);

    CHECK_EQ(view.find<StaticMeshGeometry>("obj"), mesh.get());
    CHECK_EQ(view.find<GeometryGroup>("obj"), group.get());
    CHECK_EQ(view.find<StaticMeshGeometry>("obj2"), mesh2.get());
    CHECK_EQ(view.find<GeometryGroup>("obj2"), nullptr);
}

TEST_CASE("find_all by name")
{
    SceneObjectCollection<Geometry> collection(nullptr);

    auto g0 = ref<Geometry>(new Geometry());
    g0->set_name("alpha");
    auto g1 = ref<Geometry>(new Geometry());
    g1->set_name("beta");
    auto g2 = ref<Geometry>(new Geometry());
    g2->set_name("alpha");
    collection.add(g0);
    collection.add(g1);
    collection.add(g2);

    SceneObjectCollectionView<Geometry> view(collection);

    auto alphas = view.find_all("alpha");
    REQUIRE_EQ(alphas.size(), 2);
    CHECK_EQ(alphas[0], g0.get());
    CHECK_EQ(alphas[1], g2.get());

    auto betas = view.find_all("beta");
    REQUIRE_EQ(betas.size(), 1);
    CHECK_EQ(betas[0], g1.get());

    auto gammas = view.find_all("gamma");
    CHECK(gammas.empty());
}

TEST_CASE("find_all by type")
{
    SceneObjectCollection<Geometry> collection(nullptr);

    auto mesh = ref<StaticMeshGeometry>(new StaticMeshGeometry());
    auto group = ref<GeometryGroup>(new GeometryGroup());
    auto mesh2 = ref<StaticMeshGeometry>(new StaticMeshGeometry());
    collection.add(mesh);
    collection.add(group);
    collection.add(mesh2);

    SceneObjectCollectionView<Geometry> view(collection);

    auto meshes = view.find_all<StaticMeshGeometry>();
    REQUIRE_EQ(meshes.size(), 2);
    CHECK_EQ(meshes[0], mesh.get());
    CHECK_EQ(meshes[1], mesh2.get());

    auto groups = view.find_all<GeometryGroup>();
    REQUIRE_EQ(groups.size(), 1);
    CHECK_EQ(groups[0], group.get());
}

TEST_CASE("find_all by type and name")
{
    SceneObjectCollection<Geometry> collection(nullptr);

    auto mesh = ref<StaticMeshGeometry>(new StaticMeshGeometry());
    mesh->set_name("obj");
    auto group = ref<GeometryGroup>(new GeometryGroup());
    group->set_name("obj");
    auto mesh2 = ref<StaticMeshGeometry>(new StaticMeshGeometry());
    mesh2->set_name("obj");
    collection.add(mesh);
    collection.add(group);
    collection.add(mesh2);

    SceneObjectCollectionView<Geometry> view(collection);

    auto result = view.find_all<StaticMeshGeometry>("obj");
    REQUIRE_EQ(result.size(), 2);
    CHECK_EQ(result[0], mesh.get());
    CHECK_EQ(result[1], mesh2.get());

    auto result2 = view.find_all<GeometryGroup>("obj");
    REQUIRE_EQ(result2.size(), 1);
    CHECK_EQ(result2[0], group.get());

    auto result3 = view.find_all<GeometryGroup>("missing");
    CHECK(result3.empty());
}

TEST_CASE("find on empty collection")
{
    SceneObjectCollection<Geometry> collection(nullptr);
    SceneObjectCollectionView<Geometry> view(collection);

    CHECK_EQ(view.find("anything"), nullptr);
    CHECK_EQ(view.find<StaticMeshGeometry>(), nullptr);
    CHECK(view.find_all("anything").empty());
    CHECK(view.find_all<StaticMeshGeometry>().empty());
}

TEST_SUITE_END();
