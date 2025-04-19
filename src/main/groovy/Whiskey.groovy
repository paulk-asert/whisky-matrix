/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.knowm.xchart.SwingWrapper
import se.alipsa.matrix.core.Matrix
import se.alipsa.matrix.csv.CsvImporter
import se.alipsa.matrix.stats.Correlation
import se.alipsa.matrix.xchart.HeatmapChart
import se.alipsa.matrix.xchart.RadarChart
import se.alipsa.matrix.xchart.ScatterChart
import smile.clustering.KMeans
import smile.feature.extraction.PCA

def url = getClass().getResource('whiskey.csv')
Matrix m = CsvImporter.importCsv(url).dropColumns('RowID')
println m.dimensions()
def features = m.columnNames() - 'Distillery'
def size = features.size()
features.each { feature ->
    m.apply(feature) { it.toDouble() / 4 }
}
println m.head(3)

def selected= m.subset{ it.Fruity > 0.5 && it.Sweetness > 0.5 }

println selected.dimensions()
println selected.head(10)

def aberlour = selected.subset(0..0)
def rc = RadarChart.create(aberlour)
    .addSeries('Distillery')

rc.exportPng('matrixAberlourRadar.png' as File)

rc = RadarChart.create(selected)
    .addSeries('Distillery', 80)

rc.exportPng('matrixWhiskySelectionsRadar.png' as File)
new SwingWrapper(rc.exportSwing().chart).displayChart()

def data = m.selectColumns(*features) as double[][]
def model = KMeans.fit(data,3, 20)
m['Cluster'] = model.group().toList()

def pca = PCA.fit(data)
def projected = pca.getProjection(2).apply(data)
m['X'] = projected*.getAt(0)
m['Y'] = projected*.getAt(1)

def clusters = m['Cluster'].toSet()
def sc = ScatterChart.create(m)
sc.title = 'Whiskey Flavor Clusters'
for (i in clusters) {
    def series = m.subset('Cluster', i)
    sc.addSeries("Cluster $i", series.column('X'), series.column('Y'))
}
sc.exportPng('matrixWhiskyScatterPlot.png' as File)
new SwingWrapper(sc.exportSwing().chart).displayChart()

def corr = [size<..0, 0..<size].combinations().collect { i, j ->
    Correlation.cor(data*.getAt(j), data*.getAt(i)) * 100 as int
}

def corrMatrix = Matrix.builder().data(X: 0..<corr.size(), Heat: corr)
    .types([Number] * 2)
    .matrixName('Heatmap')
    .build()

def hc = HeatmapChart.create(corrMatrix)
    .addSeries('Heat Series', features.reverse(), features, corrMatrix.column('Heat').collate(size))
hc.exportPng('matrixWhiskeyCorrHeatmap.png' as File)
new SwingWrapper(hc.exportSwing().chart).displayChart()
