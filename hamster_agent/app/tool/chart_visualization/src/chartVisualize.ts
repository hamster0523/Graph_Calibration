import path from "path";
import fs from "fs";
import puppeteer from "puppeteer";
import VMind, { ChartType, DataTable } from "@visactor/vmind";
import { isString } from "@visactor/vutils";

enum AlgorithmType {
  OverallTrending = "overallTrend",
  AbnormalTrend = "abnormalTrend",
  PearsonCorrelation = "pearsonCorrelation",
  SpearmanCorrelation = "spearmanCorrelation",
  ExtremeValue = "extremeValue",
  MajorityValue = "majorityValue",
  StatisticsAbnormal = "statisticsAbnormal",
  StatisticsBase = "statisticsBase",
  DbscanOutlier = "dbscanOutlier",
  LOFOutlier = "lofOutlier",
  TurningPoint = "turningPoint",
  PageHinkley = "pageHinkley",
  DifferenceOutlier = "differenceOutlier",
  Volatility = "volatility",
}

// 在 Node.js 环境中创建模拟的 DOM 环境
if (typeof window === 'undefined') {
  const { JSDOM } = require('jsdom');
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
  global.window = dom.window;
  global.document = dom.window.document;
  global.navigator = dom.window.navigator;
}

// const getBase64 = async (spec: any, width?: number, height?: number) => {
//   spec.animation = false;
//   width && (spec.width = width);
//   height && (spec.height = height);
//   const browser = await puppeteer.launch();
//   const page = await browser.newPage();
//   await page.setContent(getHtmlVChart(spec, width, height));

//   const dataUrl = await page.evaluate(() => {
//     const canvas: any = document
//       .getElementById("chart-container")
//       ?.querySelector("canvas");
//     return canvas?.toDataURL("image/png");
//   });

//   const base64Data = dataUrl.replace(/^data:image\/png;base64,/, "");
//   await browser.close();
//   return Buffer.from(base64Data, "base64");
// };

// 2. 修复 directory 未定义错误
// 我们需要在 getBase64 函数中添加 directory 参数
const getBase64 = async (spec: any, width?: number, height?: number, directory?: string) => {
  spec.animation = false;
  width && (spec.width = width);
  height && (spec.height = height);

  const browser = await puppeteer.launch({
    headless: true,  // 修复类型错误
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();

  // 创建临时 HTML 文件
  const htmlContent = getHtmlVChart(spec, width, height);

  // 确保 directory 有值
  const tempDir = directory || path.join(__dirname, 'temp');
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }

  const tempFilePath = path.join(tempDir, 'temp.html');
  fs.writeFileSync(tempFilePath, htmlContent);

  // 使用 file:// 协议加载本地文件
  await page.goto(`file://${tempFilePath}`, {
    waitUntil: 'networkidle0',
    timeout: 30000
  });

  // 3. 修复 waitForTimeout 方法不存在错误
  // 使用 page.waitFor() 替代 page.waitForTimeout()
  // 使用 page.waitForFunction 替代 waitFor
  await page.waitForFunction(() => {
    // 确保图表容器和 canvas 都存在
    const container = document.getElementById('chart-container');
    const canvas = container?.querySelector('canvas');

    // 检查 canvas 是否完全渲染
    return canvas && canvas.width > 0 && canvas.height > 0;
  }, {
    timeout: 30000,
    polling: 200 // 每200毫秒检查一次
  });

  const dataUrl = await page.evaluate(() => {
    const canvas: any = document
      .getElementById("chart-container")
      ?.querySelector("canvas");
    return canvas?.toDataURL("image/png");
  });

  const base64Data = dataUrl.replace(/^data:image\/png;base64,/, "");
  await browser.close();

  // 清理临时文件
  fs.unlinkSync(tempFilePath);

  return Buffer.from(base64Data, "base64");
};

const serializeSpec = (spec: any) => {
  return JSON.stringify(spec, (key, value) => {
    if (typeof value === "function") {
      const funcStr = value
        .toString()
        .replace(/(\r\n|\n|\r)/gm, "")
        .replace(/\s+/g, " ");

      return `__FUNCTION__${funcStr}`;
    }
    return value;
  });
};

// 增强 disableAllAnimations 函数
function disableAllAnimations(spec: any) {
  if (!spec) return;

  // 禁用全局动画
  spec.animation = false;
  spec.animationDuration = 0;
  spec.animationEasing = "linear";

  // 禁用组件动画
  if (spec.title) spec.title.animation = false;
  if (spec.legend) spec.legend.animation = false;
  if (spec.tooltip) spec.tooltip.animation = false;

  // 禁用系列动画
  if (spec.series && Array.isArray(spec.series)) {
    spec.series = spec.series.map((series: any) => {
      const newSeries = { ...series };
      newSeries.animation = false;
      newSeries.animationDuration = 0;
      newSeries.animationEasing = "linear";

      // 禁用内部元素动画
      if (newSeries.label) newSeries.label.animation = false;
      if (newSeries.point) newSeries.point.animation = false;
      if (newSeries.bar) newSeries.bar.animation = false;
      if (newSeries.line) newSeries.line.animation = false;
      if (newSeries.area) newSeries.area.animation = false;

      return newSeries;
    });
  }

  // 禁用坐标轴动画
  if (spec.axes && Array.isArray(spec.axes)) {
    spec.axes = spec.axes.map((axis: any) => ({
      ...axis,
      animation: false
    }));
  }

  // 禁用特殊动画配置
  if (spec.animationAppear) spec.animationAppear = false;
  if (spec.animationUpdate) spec.animationUpdate = false;
  if (spec.animationEnter) spec.animationEnter = false;
  if (spec.animationLeave) spec.animationLeave = false;

  return spec;
}


// 修改 getHtmlVChart 函数，确保安全加载
function getHtmlVChart(spec: any, width?: number, height?: number) {
  return `<!DOCTYPE html>
<html>
<head>
    <title>VChart Demo</title>
    <meta charset="UTF-8">
    <script src="https://unpkg.com/@visactor/vchart@1.7.2/build/index.min.js"></script>
    <style>
      body, html {
        margin: 0;
        padding: 0;
        overflow: hidden;
        width: 100%;
        height: 100%;
      }
      #chart-container {
        width: ${width ? `${width}px` : "100%"};
        height: ${height ? `${height}px` : "100%"};
      }
    </style>
</head>
<body>
    <div id="chart-container"></div>
    <script>
      // 防止未定义错误
      if (typeof VChart === 'undefined') {
        console.error('VChart is not loaded');
      } else {
        // 安全解析函数
        function parseSpec(stringSpec) {
          try {
            return JSON.parse(stringSpec, (k, v) => {
              if (typeof v === 'string' && v.startsWith('__FUNCTION__')) {
                try {
                  // 安全创建函数
                  return Function('"use strict"; return (' + v.slice(12) + ')')();
                } catch(e) {
                  console.warn('Function parse error:', e);
                  return () => {};
                }
              }
              return v;
            });
          } catch (e) {
            console.error('Spec parse error:', e);
            return {};
          }
        }

        const spec = parseSpec(\`${serializeSpec(spec)}\`);

        // 双重禁用动画
        if (spec) {
          spec.animation = false;
          if (spec.series) {
            spec.series.forEach(s => s.animation = false);
          }
        }

        const container = document.getElementById('chart-container');
        if (container) {
          const chart = new VChart.VChart(spec, {
            dom: container,
            disableDirtyRect: true, // 提高渲染性能
            renderer: 'canvas' // 强制使用 canvas 渲染
          });

          // 添加渲染完成检测
          let renderCompleted = false;
          chart.on('rendered', () => {
            renderCompleted = true;
          });

          chart.renderAsync().catch(e => {
            console.error('Chart render error:', e);
          });

          // 暴露状态供 Puppeteer 检测
          window.chartRenderCompleted = () => renderCompleted;
        } else {
          console.error('Chart container not found');
        }
      }
    </script>
</body>
</html>`;
}

/**
 * get file path saved string
 * @param isUpdate {boolean} default: false, update existed file when is true
 */
function getSavedPathName(
  directory: string,
  fileName: string,
  outputType: "html" | "png" | "json" | "md",
  isUpdate: boolean = false
) {
  let newFileName = fileName;
  while (
    !isUpdate &&
    fs.existsSync(
      path.join(directory, "visualization", `${newFileName}.${outputType}`)
    )
  ) {
    newFileName += "_new";
  }
  return path.join(directory, "visualization", `${newFileName}.${outputType}`);
}

const readStdin = (): Promise<string> => {
  return new Promise((resolve) => {
    let input = "";
    process.stdin.setEncoding("utf-8"); // 确保编码与 Python 端一致
    process.stdin.on("data", (chunk) => (input += chunk));
    process.stdin.on("end", () => resolve(input));
  });
};

/** Save insights markdown in local, and return content && path */
const setInsightTemplate = (
  path: string,
  title: string,
  insights: string[]
) => {
  let res = "";
  if (insights.length) {
    res += `## ${title} Insights`;
    insights.forEach((insight, index) => {
      res += `\n${index + 1}. ${insight}`;
    });
  }
  if (res) {
    fs.writeFileSync(path, res, "utf-8");
    return { insight_path: path, insight_md: res };
  }
  return {};
};

/** Save vmind result into local file, Return chart file path */
async function saveChartRes(options: {
  spec: any;
  directory: string;
  outputType: "png" | "html";
  fileName: string;
  width?: number;
  height?: number;
  isUpdate?: boolean;
}) {
  const { directory, fileName, spec, outputType, width, height, isUpdate } =
    options;
  const specPath = getSavedPathName(directory, fileName, "json", isUpdate);
  fs.writeFileSync(specPath, JSON.stringify(spec, null, 2));
  const savedPath = getSavedPathName(directory, fileName, outputType, isUpdate);
  if (outputType === "png") {
    // 传递 directory 参数
    const base64 = await getBase64(spec, width, height, directory);
    fs.writeFileSync(savedPath, base64);
  } else {
    const html = getHtmlVChart(spec, width, height);
    fs.writeFileSync(savedPath, html, "utf-8");
  }
  return savedPath;
}

async function generateChart(
  vmind: VMind,
  options: {
    dataset: string | DataTable;
    userPrompt: string;
    directory: string;
    outputType: "png" | "html";
    fileName: string;
    width?: number;
    height?: number;
    language?: "en" | "zh";
  }
) {
  let res: {
    chart_path?: string;
    error?: string;
    insight_path?: string;
    insight_md?: string;
  } = {};
  const {
    dataset,
    userPrompt,
    directory,
    width,
    height,
    outputType,
    fileName,
    language,
  } = options;
  try {
    // Get chart spec and save in local file
    const jsonDataset = isString(dataset) ? JSON.parse(dataset) : dataset;
    const { spec, error, chartType } = await vmind.generateChart(
      userPrompt,
      undefined,
      jsonDataset,
      {
        enableDataQuery: false,
        theme: "light",
      }
    );
    if (error || !spec) {
      return {
        error: error || "Spec of Chart was Empty!",
      };
    }

    disableAllAnimations(spec);

    spec.title = {
      text: userPrompt,
    };
    if (!fs.existsSync(path.join(directory, "visualization"))) {
      fs.mkdirSync(path.join(directory, "visualization"));
    }
    const specPath = getSavedPathName(directory, fileName, "json");
    res.chart_path = await saveChartRes({
      directory,
      spec,
      width,
      height,
      fileName,
      outputType,
    });

    // get chart insights and save in local
    const insights = [];
    if (
      chartType &&
      [
        ChartType.BarChart,
        ChartType.LineChart,
        ChartType.AreaChart,
        ChartType.ScatterPlot,
        ChartType.DualAxisChart,
      ].includes(chartType)
    ) {
      const { insights: vmindInsights } = await vmind.getInsights(spec, {
        maxNum: 6,
        algorithms: [
          AlgorithmType.OverallTrending,
          AlgorithmType.AbnormalTrend,
          AlgorithmType.PearsonCorrelation,
          AlgorithmType.SpearmanCorrelation,
          AlgorithmType.StatisticsAbnormal,
          AlgorithmType.LOFOutlier,
          AlgorithmType.DbscanOutlier,
          AlgorithmType.MajorityValue,
          AlgorithmType.PageHinkley,
          AlgorithmType.TurningPoint,
          AlgorithmType.StatisticsBase,
          AlgorithmType.Volatility,
        ],
        usePolish: false,
        language: language === "en" ? "english" : "chinese",
      });
      insights.push(...vmindInsights);
    }
    const insightsText = insights
      .map((insight) => insight.textContent?.plainText)
      .filter((insight) => !!insight) as string[];
    spec.insights = insights;
    fs.writeFileSync(specPath, JSON.stringify(spec, null, 2));
    res = {
      ...res,
      ...setInsightTemplate(
        getSavedPathName(directory, fileName, "md"),
        userPrompt,
        insightsText
      ),
    };
  } catch (error: any) {
    res.error = error.toString();
  } finally {
    return res;
  }
}

async function updateChartWithInsight(
  vmind: VMind,
  options: {
    directory: string;
    outputType: "png" | "html";
    fileName: string;
    insightsId: number[];
  }
) {
  const { directory, outputType, fileName, insightsId } = options;
  let res: { error?: string; chart_path?: string } = {};
  try {
    const specPath = getSavedPathName(directory, fileName, "json", true);
    const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));
    // llm select index from 1
    const insights = (spec.insights || []).filter(
      (_insight: any, index: number) => insightsId.includes(index + 1)
    );
    const { newSpec, error } = await vmind.updateSpecByInsights(spec, insights);
    if (error) {
      throw error;
    }
    res.chart_path = await saveChartRes({
      spec: newSpec,
      directory,
      outputType,
      fileName,
      isUpdate: true,
    });
  } catch (error: any) {
    res.error = error.toString();
  } finally {
    return res;
  }
}

async function executeVMind() {
  const input = await readStdin();
  const inputData = JSON.parse(input);
  let res;
  const {
    llm_config,
    width,
    dataset = [],
    height,
    directory,
    user_prompt: userPrompt,
    output_type: outputType = "png",
    file_name: fileName,
    task_type: taskType = "visualization",
    insights_id: insightsId = [],
    language = "en",
  } = inputData;
  const { base_url: baseUrl, model, api_key: apiKey } = llm_config;
  const vmind = new VMind({
    url: `${baseUrl}/chat/completions`,
    model,
    headers: {
      "api-key": apiKey,
      Authorization: `Bearer ${apiKey}`,
    },
  });
  if (taskType === "visualization") {
    res = await generateChart(vmind, {
      dataset,
      userPrompt,
      directory,
      outputType,
      fileName,
      width,
      height,
      language,
    });
  } else if (taskType === "insight" && insightsId.length) {
    res = await updateChartWithInsight(vmind, {
      directory,
      fileName,
      outputType,
      insightsId,
    });
  }
  console.log(JSON.stringify(res));
}

executeVMind();
