window.__ModuleLoader__.load({
  id: 'lingming-memory',
  factory(require) {
    const React = require('react');
    const h = React.createElement;

    // ============ 官方配置表单（configForms + plugins.bundle.config） ============
    const FIELDS = [
      { key: 'enabled', label: '总开关', hint: '是否启用灵铭记忆（也可用 /lingming 切换）', kind: 'bool', def: true },
      { key: 'embedServerUrl', label: '服务地址', hint: '本地记忆服务地址', kind: 'text', def: 'http://127.0.0.1:9000' },
      { key: 'python', label: 'Python 路径', hint: '带 torch + sentence-transformers 的 Python', kind: 'text', def: 'D:\\conda\\python.exe' },
      { key: 'topK', label: '召回条数', hint: '每轮召回的历史记忆条数', kind: 'number', def: 5 },
      { key: 'inject', label: '注入上下文', hint: '关掉就只存不用', kind: 'bool', def: true },
      { key: 'minScore', label: '相关性阈值', hint: '低于它不注入（实测无关 0.30~0.37，相关 0.56~0.72）', kind: 'number', def: 0.45 },
      { key: 'contextMode', label: '上下文模式', hint: 'augment=历史+记忆 / window=最近几条+记忆 / replace=仅当前+记忆', kind: 'text', def: 'augment' },
      { key: 'keepMessages', label: '保留原始条数', hint: 'window 模式下保留最近几条原始消息', kind: 'number', def: 6 },
      { key: 'historyTurns', label: '拼接轮数', hint: '短查询用最近几轮补语义', kind: 'number', def: 3 },
      { key: 'cMult', label: '坐标桶倍数', hint: '桶数 C = 倍数 × 编码维度（en=384 / zh=512）', kind: 'number', def: 4 },
      { key: 'k', label: '稀疏 top-k', hint: '每条记忆点亮几个桶（C=2048 时 k≥16）', kind: 'number', def: 16 },
      { key: 'capacity', label: '记忆容量', hint: '最多存多少条（超过自动扩容）', kind: 'number', def: 10000 },
      { key: 'writeScale', label: '写入强度', hint: '外积写 g 的步长', kind: 'number', def: 0.1 },
      { key: 'gateBias', label: '写门控偏置', hint: '越小越倾向写入', kind: 'number', def: 0.2 },
      { key: 'gateTemp', label: '写门控温度', hint: '越小门控越硬', kind: 'number', def: 0.05 },
      { key: 'readTop', label: '读取聚合数', hint: '召回时取最相关几条求平均', kind: 'number', def: 3 },
    ];

    const rowStyle = { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', padding: '6px 0' };
    const labelStyle = { display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0 };
    const titleStyle = { fontSize: '13px', color: 'var(--dsw-alias-label-primary, inherit)' };
    const hintStyle = { fontSize: '12px', color: 'var(--dsw-alias-label-secondary, rgba(127,127,127,.8))' };
    const inputStyle = { width: '200px', padding: '4px 8px', borderRadius: '6px', border: '1px solid var(--dsw-alias-border, #888)', background: 'transparent', color: 'inherit', fontSize: '13px' };

    function Field(props) {
      const { f, value, writable, scope } = props;
      const current = value[f.key] !== undefined ? value[f.key] : f.def;
      const commit = (val) => { try { if (writable && scope && typeof scope.set === 'function') scope.set(f.key, val); } catch (e) {} };
      let control;
      if (f.kind === 'bool') {
        control = h('input', { type: 'checkbox', checked: current === true, disabled: !writable, onChange: (e) => commit(e.target.checked) });
      } else if (f.kind === 'number') {
        control = h('input', { type: 'number', value: current === undefined ? '' : current, style: inputStyle, disabled: !writable, onBlur: (e) => { const n = Number(e.target.value); if (!Number.isNaN(n)) commit(n); } });
      } else {
        control = h('input', { type: 'text', value: current === undefined ? '' : current, style: inputStyle, disabled: !writable, onBlur: (e) => commit(e.target.value) });
      }
      return h('div', { style: rowStyle },
        h('div', { style: labelStyle }, h('span', { style: titleStyle }, f.label), h('span', { style: hintStyle }, f.hint)),
        control,
      );
    }

    function LingmingConfig(props) {
      if (props.view !== 'page') return null;
      const scope = props.scope;
      if (!scope) return null;
      const snapshot = React.useSyncExternalStore(
        (listener) => scope.subscribe(listener),
        () => scope.getSnapshot(),
      );
      const value = snapshot.value || {};
      const writable = snapshot.writable === true;
      const resetDefaults = () => {
        try { if (writable && typeof scope.unset === 'function') for (const f of FIELDS) scope.unset(f.key); } catch (e) {}
      };
      return h('div', { style: { display: 'grid', gap: '2px', padding: '4px 0' } },
        ...FIELDS.map((f) => h(Field, { key: f.key, f, value, writable, scope })),
        h('div', { style: { display: 'flex', gap: '10px', marginTop: '10px' } },
          h('button', {
            type: 'button',
            onClick: resetDefaults,
            disabled: !writable,
            style: { padding: '5px 14px', borderRadius: '6px', border: '1px solid var(--dsw-alias-border, #888)', background: 'transparent', color: 'inherit', fontSize: '13px', cursor: 'pointer' },
          }, '恢复默认'),
        ),
      );
    }

    // 官方表单：configForms 服务取本插件表单（namespace = row id），挂到 plugins.bundle.config
    function mountConfigForm(ctx) {
      try {
        const forms = typeof ctx.get === 'function' ? ctx.get('configForms') : undefined;
        if (!forms || typeof forms.get !== 'function') return;
        let scope;
        for (const ns of ['lingming-memory', 'include:lingming-memory']) {
          try {
            const f = forms.get(ns);
            if (f && typeof f.getSnapshot === 'function') { scope = f; break; }
          } catch (e) {}
        }
        if (!scope) return;
        ctx.slots.inject('plugins.bundle.config', () => ctx.slots.register(
          { name: 'plugins.bundle.config', key: 'lingming-memory', inject: () => ({ scope }) },
          LingmingConfig,
        ));
      } catch (e) {}
    }

    return {
      inject: ['slots'],
      apply(ctx) {
        // 底部开关（之前一直触发 source 错误，先移除；配置表单保留）
        mountConfigForm(ctx);
      },
    };
  },
});
