import asyncio
from playwright.async_api import async_playwright
import os
import csv

async def scrape_douyin_all(limit=2):
    async with async_playwright() as p:
        # 修改成本地浏览器路径（Edge示例，可换成Chrome路径）
        browser_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        user_data_dir = os.path.expanduser("~/.playwright-chrome-user")

        # 启动本地浏览器（持久化上下文，保留登录态）
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            executable_path=browser_path,
            headless=False
        )

        page = await browser.new_page()
        await page.goto("https://www.douyin.com/?recommend=1")

        results = []
        seen = set()

        while len(results) < limit:
            try:
                # 点赞
                likes = await page.query_selector(".KV_gO8oI")
                likes_text = await likes.inner_text() if likes else ""

                # # 评论
                # comments = await page.query_selector(".X_wB9MpJ")
                # comments_text = await comments.inner_text() if comments else ""
                #发布时间
                publish_time = await page.query_selector(".time")
                publish_time = await publish_time.inner_text() if publish_time else ""    

                # 收藏
                favorites = await page.query_selector(".OjAuUiYV")
                favorites_text = await favorites.inner_text() if favorites else ""

                # 转发
                reposts = await page.query_selector(".hzIYk71v")
                reposts_text = await reposts.inner_text() if reposts else ""

                # 时长
                duration = await page.query_selector(".time-duration")
                duration_text = await duration.inner_text() if duration else ""

                # 唯一 key
                key = (likes_text, publish_time, favorites_text, reposts_text, duration_text)
                if key not in seen and any(key):  # 至少有一项不为空
                    seen.add(key)
                    results.append(key)
                    print(f"[{len(results)}] 发布时间:{publish_time},时长:{duration_text},点赞:{likes_text},收藏:{favorites_text},转发:{reposts_text} ")

            except Exception as e:
                print("采集出错:", e)

            # 模拟刷视频
            await page.keyboard.press("ArrowDown")
            await page.wait_for_timeout(1500)

        print(f"采集完成，共获取 {len(results)} 条视频数据")

        # 保存到 CSV
        with open("Data_Soure/douyin_data_source2.csv", "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "likes", "publish_time", "favorites", "reposts", "duration"])
            for i, (likes, publish_time, favorites, reposts, duration) in enumerate(results, start=1):
                writer.writerow([i, likes, publish_time, favorites, reposts, duration])

        print("结果已保存到 douyin_data_source2.csv")
        await browser.close()
        return results


if __name__ == "__main__":
    asyncio.run(scrape_douyin_all(limit=2))
