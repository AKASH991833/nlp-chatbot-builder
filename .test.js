module.exports.run = async (page, base, shot) => {
  const out = [];
  await page.waitForSelector('.msg.bot', {timeout: 8000});
  const intents = await page.$$eval('.intent', r => r.length);
  out.push(['5 default intents', intents === 5]);
  const acc = await page.$eval('#m-acc', e => e.textContent);
  out.push(['holdout acc rendered: ' + acc, /%|–/.test(acc)]);
  // chat: ask about hours
  await page.type('#chatIn', 'when are you open?');
  await page.click('#send');
  await page.waitForFunction(() => document.querySelectorAll('.msg.bot').length >= 2, {timeout: 8000});
  await new Promise(r=>setTimeout(r,500));
  const last = await page.$$eval('.msg.bot', m => m[m.length-1].textContent);
  out.push(['bot answered hours intent: ' + last.slice(0,50), /9 AM|8 PM/.test(last)]);
  const conf = await page.$$eval('.msg.bot .conf', m => m.length ? m[m.length-1].textContent : '');
  out.push(['confidence shown: ' + conf, /confidence \d+%/.test(conf)]);
  // unknown message fallback
  await page.type('#chatIn', 'xyzzy quantum banana');
  await page.click('#send');
  await new Promise(r=>setTimeout(r,600));
  const last2 = await page.$$eval('.msg.bot', m => m[m.length-1].textContent);
  out.push(['fallback on unknown', /not sure/i.test(last2)]);
  // add an intent via prompt dialog
  page.on('dialog', d => d.accept('jokes'));
  await page.click('#addIntent');
  await new Promise(r=>setTimeout(r,300));
  const n2 = await page.$$eval('.intent', r => r.length);
  out.push(['intent added', n2 === 6]);
  if (shot) await page.screenshot({path: shot});
  return out;
};
