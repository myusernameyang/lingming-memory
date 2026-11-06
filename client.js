window.__ModuleLoader__.load({
  id: 'lingming-memory',
  factory(require) {
    const React = require('react');
    const h = React.createElement;

    // 最小开关：纯本地状态，不调 remote，不注册配置表单
    function Toggle() {
      const [on, setOn] = React.useState(false);
      return h('button', {
        onClick: () => setOn((v) => !v),
        role: 'switch',
        'aria-checked': on,
        title: '灵铭记忆',
        style: {
          padding: '3px 10px', borderRadius: '999px',
          border: '1px solid var(--dsw-alias-border, #888)',
          background: on ? 'var(--dsw-alias-primary, #247bbf)' : 'transparent',
          color: on ? '#fff' : 'inherit',
          fontSize: '12px', cursor: 'pointer', lineHeight: '20px',
        },
      }, on ? '灵铭·开' : '灵铭·关');
    }

    return {
      inject: ['slots'],
      apply(ctx) {
        ctx.slots.inject('conversation.composer.dock', () => ctx.slots.register(
          { name: 'conversation.composer.dock', id: 'lingming-memory', order: 5 },
          Toggle,
        ));
      },
    };
  },
});
