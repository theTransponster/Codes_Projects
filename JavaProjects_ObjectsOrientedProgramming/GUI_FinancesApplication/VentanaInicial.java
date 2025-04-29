import javax.swing.JFrame;

public class VentanaInicial extends JFrame {

	public VentanaInicial(){
		super("Ventana Inicial para cálculo de ISR");
		this.setDefaultCloseOperation(EXIT_ON_CLOSE);
		this.setLocation(400, 100);
		PanelInicial p=new PanelInicial();
		this.add(p);
		this.pack();
		this.setVisible(true);
		
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		VentanaInicial v=new VentanaInicial();
	}

}
